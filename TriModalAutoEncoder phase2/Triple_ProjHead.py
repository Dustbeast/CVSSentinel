"""
Feature Extraction — Approach 2 (v4): Triple-Projection Fusion Autoencoder
===========================================================================
Architecture:
  Three independent projection heads give equal capacity to each modality
  before fusion, preventing any single stream from dominating gradients.

  Modalities:
    Stream A — Semantic  : SecureBERT embeddings        768-dim
    Stream B — Structural: CVSS ordinal + base score      9-dim
    Stream C — CWE       : SVD-compressed multi-hot      64-dim

  proj_sem  : 768 → 256  (z_sem)
  proj_str  :   9 → 256  (z_str)
  proj_cwe  :  64 → 256  (z_cwe)
  mlp       : 768 → 256  (fused bottleneck — concat of three 256-dim projections)
  dec_sem   : 256 → 768
  dec_str   : 256 →   9
  dec_cwe   : 256 →  64

  Training loss (mean reduction balances all three decoders automatically):
    L = MSE(recon_sem, sem) + MSE(recon_str, str) + MSE(recon_cwe, cwe)

  Output node features:
    [fused(256)]                        → DOMINANT, CoLA
    [sem(768) | str(9) | cwe_svd(64)]  → AnomalyDAE (same info, un-fused)

Output files (versioned v4):
  node_features_v4.pt              — [N, 256]  fused, DOMINANT + CoLA
  node_features_anomalydae_v4.pt   — [N, 841]  raw, AnomalyDAE control
  triple_projection_model_v4.pt    — trained autoencoder weights
  cve_id_mapping_v4.csv            — CVE_ID to row-index mapping
  reconstruction_diagnostics_v4.pt — [N, 10]   interpretability only
  cwe_svd_components_v4.npy        — SVD components for CWE interpretability
  cwe_classes_v4.csv               — CWE label index for interpretability
"""

import os
import logging
from huggingface_hub import eval_result_entries_to_yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CVSS_ORDINAL_MAPS = {
    "AV": {"N": 3, "A": 2, "L": 1, "P": 0},
    "AC": {"L": 1, "H": 0},
    "PR": {"N": 2, "L": 1, "H": 0},
    "UI": {"N": 1, "R": 0},
    "S":  {"C": 1, "U": 0},
    "C":  {"H": 2, "L": 1, "N": 0},
    "I":  {"H": 2, "L": 1, "N": 0},
    "A":  {"H": 2, "L": 1, "N": 0},
}
METRIC_ORDER   = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

# NVD placeholder CWEs — carry no vulnerability class information
FAKE_CWES      = {"NVD-CWE-noinfo", "NVD-CWE-Other"}

SEMANTIC_DIM   = 768
STRUCTURAL_DIM = 9      # 8 ordinal metrics + 1 base score
CWE_SVD_DIM    = 64     # compressed CWE representation
PROJ_DIM       = 256
FUSED_DIM      = 256
ANOMALYDAE_DIM = SEMANTIC_DIM + STRUCTURAL_DIM + CWE_SVD_DIM   # 841


# ===========================================================================
# MODEL
# ===========================================================================

class TripleProjectionFusionAE(nn.Module):
    """
    Triple-Projection Fusion Autoencoder (Approach 2, v4).

    Three independent projection heads map each modality to an equal
    256-dim space before fusion. With mean-reduction MSE loss, all three
    decoders receive balanced gradient flow — no modality dominates.

    CWE is a first-class modality here, not appended after training.
    Its 64-dim SVD representation is projected, fused, and decoded
    alongside semantic and structural streams from the start.
    """

    def __init__(
        self,
        semantic_dim:    int   = SEMANTIC_DIM,
        structural_dim:  int   = STRUCTURAL_DIM,
        cwe_dim:         int   = CWE_SVD_DIM,
        proj_dim:        int   = PROJ_DIM,
        dropout:         float = 0.3,
    ):
        super().__init__()
        self.semantic_dim   = semantic_dim
        self.structural_dim = structural_dim
        self.cwe_dim        = cwe_dim
        self.proj_dim       = proj_dim

        # ── Three independent projection heads ─────────────────────────────
        # Each modality gets its own path to the shared 256-dim space.
        # Architectural guarantee against gradient imbalance — a 768-dim
        # stream cannot crowd out a 9-dim stream because they never share
        # linear layers until after projection.
        self.proj_sem = nn.Sequential(
            nn.Linear(semantic_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
            
        )
        self.proj_str = nn.Sequential(
            nn.Linear(structural_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
            

        )
        self.proj_cwe = nn.Sequential(
            nn.Linear(cwe_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)

        )

        # ── Shared MLP bottleneck ───────────────────────────────────────────
        # Input : concat of three 256-dim projections = 768-dim
        # Output: 256-dim fused representation
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim * 2),   # 768 → 512
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),        # 512 → 256
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )

        # ── Three independent decoders ──────────────────────────────────────
        # Each reconstructs its own original modality from the 256-dim
        # fused bottleneck. Gradient flows back through all three
        # projection heads simultaneously.
        # Deepened semantic decoder — handles 3× expansion more gracefully
        # LayerNorm at mid-point stabilizes the steepest expansion
        self.dec_sem = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 2),        # 256 → 512
            nn.ReLU(),
            nn.LayerNorm(proj_dim*2),
            nn.Linear(proj_dim*2,proj_dim*3),
            nn.ReLU(),
            nn.Linear(proj_dim * 3, semantic_dim),    # 512 → 768
        )
        self.dec_str = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),       # 256 → 128
            nn.ReLU(),
            nn.Linear(proj_dim // 2, structural_dim), # 128 →   9
        )
        self.dec_cwe = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),       # 256 → 128
            nn.ReLU(),
            nn.Linear(proj_dim // 2, cwe_dim),        # 128 →  64
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x_sem: torch.Tensor,   # [N, 768]
        x_str: torch.Tensor,   # [N,   9]
        x_cwe: torch.Tensor,   # [N,  64]
    ):
        z_sem = self.proj_sem(x_sem)   # [N, 256]
        z_str = self.proj_str(x_str)   # [N, 256]
        z_cwe = self.proj_cwe(x_cwe)   # [N, 256]

        fused = self.mlp(
            torch.cat([z_sem, z_str, z_cwe], dim=1)   # [N, 768]
        )   # [N, 256] --Layernorm Applied

        recon_sem = self.dec_sem(fused)   # [N, 768]
        recon_str = self.dec_str(fused)   # [N,   9]
        recon_cwe = self.dec_cwe(fused)   # [N,  64]

        return fused, recon_sem, recon_str, recon_cwe


# ===========================================================================
# EXTRACTOR
# ===========================================================================

class TripleProjectionFusionExtractor:
    """
    End-to-end pipeline for Approach 2 (v4) feature extraction.

    Steps:
        1. Robust CSV ingestion, sort by CVE_ID
        2. SecureBERT embeddings  — semantic stream      [N, 768]
        3. CVSS ordinal encoding  — structural stream    [N,   9]
        4. CWE multi-hot + SVD    — CWE stream           [N,  64]
        5. L2-normalize all three streams independently
        6. Train TripleProjectionFusionAE
        7. Extract 256-dim fused features and 841-dim raw AnomalyDAE features
        8. Save all outputs versioned v4
    """

    def __init__(
        self,
        nodes_path:     str = "nodes_cve.csv",
        cwe_edges_path: str = "edges_cve_cwe.csv",
        model_name:     str = "cisco-ai/SecureBERT2.0-biencoder",
    ):
        self.nodes_path     = nodes_path
        self.cwe_edges_path = cwe_edges_path
        self.model_name     = model_name

        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_torch = torch.device(self.device)
        log.info(f"Device: {self.device}")

        if not os.path.exists(self.nodes_path):
            raise FileNotFoundError(f"Node file not found: {self.nodes_path}")
        if not os.path.exists(self.cwe_edges_path):
            raise FileNotFoundError(f"CWE edges not found: {self.cwe_edges_path}")

        # ── Robust CSV ingestion ────────────────────────────────────────────
        try:
            self.df = pd.read_csv(
                self.nodes_path,
                engine="python",
                quotechar='"',
                on_bad_lines="warn",
            )
        except Exception:
            log.warning("Primary CSV load failed — retrying with on_bad_lines='skip'")
            self.df = pd.read_csv(self.nodes_path, on_bad_lines="skip")

        # Sort by CVE_ID — every output file must share this ordering
        self.df = self.df.sort_values("CVE_ID").reset_index(drop=True)

        self.df["Description"] = self.df["Description"].fillna(
            "No description available"
        )
        self.df["CVSS_Vector"] = self.df["CVSS_Vector"].fillna("")

        if "CVSS_BaseScore" in self.df.columns:
            self.df["CVSS_BaseScore"] = pd.to_numeric(
                self.df["CVSS_BaseScore"], errors="coerce"
            ).fillna(0.0)
        else:
            log.warning(
                "CVSS_BaseScore column not found — defaulting to 0.0. "
                "Check your CSV column names."
            )
            self.df["CVSS_BaseScore"] = 0.0

        log.info(f"Loaded {len(self.df):,} CVEs from {self.nodes_path}")

    # -----------------------------------------------------------------------
    # STEP 1 — Semantic embeddings
    # -----------------------------------------------------------------------
    def _get_semantic_embeddings(self) -> np.ndarray:
        log.info("─" * 60)
        log.info("STEP 1 — Semantic embeddings (SecureBERT, max_seq=256)")
        log.info("─" * 60)

        model = SentenceTransformer(self.model_name, device=self.device)
        # 256 tokens is the Goldilocks zone for CVE text:
        #   < 128 clips impact statements at end of paragraph
        #   > 256 dilutes the 768-dim vector with version-list boilerplate
        #   and quadruples O(N^2) attention compute for marginal gain
        model.max_seq_length = 256

        if self.device == "cuda":
            model.half()
            log.info("FP16 enabled for GPU inference")

        embeddings = model.encode(
            self.df["Description"].tolist(),
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,   # L2-normalize manually below
        )

        embeddings = normalize(embeddings, norm="l2")
        log.info(f"Semantic embeddings : {embeddings.shape}  (L2-normalized)")
        return embeddings.astype(np.float32)

    # -----------------------------------------------------------------------
    # STEP 2 — Structural features
    # -----------------------------------------------------------------------
    def _parse_cvss_vector(self, vec_str: str) -> dict:
        if not isinstance(vec_str, str) or not vec_str:
            return {}
        result = {}
        for part in vec_str.split("/"):
            if ":" in part:
                k, v = part.split(":", 1)
                result[k] = v
        return result

    def _get_structural_features(self) -> np.ndarray:
        log.info("─" * 60)
        log.info("STEP 2 — Structural features (CVSS ordinal + base score)")
        log.info("─" * 60)

        rows      = []
        n_missing = 0

        for _, row in self.df.iterrows():
            parsed  = self._parse_cvss_vector(row["CVSS_Vector"])
            encoded = []

            for metric in METRIC_ORDER:
                val_str = parsed.get(metric, None)
                mapping = CVSS_ORDINAL_MAPS[metric]
                if val_str and val_str in mapping:
                    encoded.append(float(mapping[val_str]))
                else:
                    encoded.append(float(max(mapping.values())) / 2.0)
                    n_missing += 1

            encoded.append(float(row["CVSS_BaseScore"]) / 10.0)
            rows.append(encoded)

        if n_missing > 0:
            log.warning(f"{n_missing:,} metric values imputed with midpoint")

        features = np.array(rows, dtype=np.float32)
        features = normalize(features, norm="l2")
        log.info(f"Structural features : {features.shape}  (L2-normalized)")
        return features

    # -----------------------------------------------------------------------
    # STEP 3 — CWE features
    # -----------------------------------------------------------------------
    def _get_cwe_features(self) -> tuple:
        """
        Build a 64-dim L2-normalized CWE representation per CVE.

        Pipeline:
          1. Load CVE→CWE edges, strip NVD placeholder CWEs
          2. Multi-hot encode real CWEs aligned to df row order
          3. TruncatedSVD: 705-dim sparse → 64-dim dense
          4. L2-normalize rows

        Returns (cwe_dense, svd, mlb):
          cwe_dense : np.ndarray [N, 64]
          svd       : fitted TruncatedSVD — saved for interpretability
          mlb       : fitted MultiLabelBinarizer — saved for interpretability

        CVEs whose only CWE was a placeholder receive a zero vector.
        This is correct — they have no real vulnerability class signal.
        """
        log.info("─" * 60)
        log.info("STEP 3 — CWE features (multi-hot → SVD-64 → L2-normalize)")
        log.info("─" * 60)

        edges_cwe = pd.read_csv(self.cwe_edges_path)
        log.info(f"CWE edges loaded         : {len(edges_cwe):,}")

        # Strip NVD placeholders
        edges_cwe_clean = edges_cwe[~edges_cwe["Weakness_CWE"].isin(FAKE_CWES)]
        removed = len(edges_cwe) - len(edges_cwe_clean)
        log.info(f"Placeholder edges removed: {removed:,}  ({FAKE_CWES})")
        log.info(f"Real CWE edges           : {len(edges_cwe_clean):,}")

        # Group CWEs per CVE
        cve_cwes = (
            edges_cwe_clean
            .groupby("CVE_ID")["Weakness_CWE"]
            .apply(list)
            .reset_index()
        )

        # Align to df row order via left merge on sorted CVE_ID
        mapping_df = self.df[["CVE_ID"]].copy()
        cve_cwes   = mapping_df.merge(cve_cwes, on="CVE_ID", how="left")
        cve_cwes["Weakness_CWE"] = cve_cwes["Weakness_CWE"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        no_real_cwe = (cve_cwes["Weakness_CWE"].apply(len) == 0).sum()
        log.info(
            f"CVEs with no real CWE    : {no_real_cwe:,}  "
            f"(zero CWE vector assigned)"
        )

        # Multi-hot encode
        mlb          = MultiLabelBinarizer()
        cwe_multihot = mlb.fit_transform(cve_cwes["Weakness_CWE"])
        log.info(
            f"Multi-hot shape          : {cwe_multihot.shape}  "
            f"sparsity={1 - cwe_multihot.mean():.4f}"
        )

        # SVD compression
        svd       = TruncatedSVD(n_components=CWE_SVD_DIM, random_state=42)
        cwe_dense = svd.fit_transform(cwe_multihot).astype(np.float32)
        evr = svd.explained_variance_ratio_.sum()
        if evr < 0.70:
            log.warning(f"SVD explains only {evr:.2%} variance — consider reducing CWE_SVD_DIM or checking CWE coverage")
        log.info(
            f"SVD explained variance   : {evr:.4f}"
        )

        # L2-normalize — same treatment as all other streams
        cwe_dense = normalize(cwe_dense, norm="l2")
        log.info(f"CWE features             : {cwe_dense.shape}  (L2-normalized)")

        return cwe_dense, svd, mlb

    # -----------------------------------------------------------------------
    # STEP 4 — Train autoencoder
    # -----------------------------------------------------------------------
    def _train_autoencoder(
        self,
        sem:        np.ndarray,
        str_:       np.ndarray,
        cwe:        np.ndarray,
        epochs:     int   = 500,
        lr:         float = 1e-3,
        batch_size: int   = 1024,
        patience:   int   = 30,
    ) -> TripleProjectionFusionAE:
        log.info("─" * 60)
        log.info("STEP 4 — Training TripleProjectionFusionAE")
        log.info("─" * 60)
        log.info(
            f"epochs={epochs}  lr={lr}  "
            f"batch_size={batch_size}  patience={patience}"
        )

        x_sem = torch.from_numpy(sem).to(self.device_torch)
        x_str = torch.from_numpy(str_).to(self.device_torch)
        x_cwe = torch.from_numpy(cwe).to(self.device_torch)
        n     = x_sem.size(0)

        model = TripleProjectionFusionAE(
            semantic_dim   = sem.shape[1],
            structural_dim = str_.shape[1],
            cwe_dim        = cwe.shape[1],
            proj_dim       = PROJ_DIM,
            dropout        = 0.3,
        ).to(self.device_torch)

        total_params = sum(p.numel() for p in model.parameters())
        log.info(f"Model parameters         : {total_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )

        loss_history = []
        loss_sem_history = []
        loss_str_history = []
        loss_cwe_history = []
        lr_history = []

        best_loss    = float("inf")
        patience_ctr = 0
        best_state   = None
        best_epoch   = 1 

        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss  = 0.0
            epoch_loss_sem = 0.0
            epoch_loss_str = 0.0
            epoch_loss_cwe = 0.0
            num_batches = 0
            perm        = torch.randperm(n, device=self.device_torch)

            for start in range(0, n, batch_size):
                idx   = perm[start : start + batch_size]
                b_sem = x_sem[idx]
                b_str = x_str[idx]
                b_cwe = x_cwe[idx]

                optimizer.zero_grad()
                fused, recon_sem, recon_str, recon_cwe = model(
                    b_sem, b_str, b_cwe
                )

                # Three-term balanced loss.
                # Mean reduction normalizes each term per element — no
                # manual weighting needed across the three decoder branches.
                loss_sem = F.mse_loss(recon_sem, b_sem)
                loss_str = F.mse_loss(recon_str, b_str)
                loss_cwe = F.mse_loss(recon_cwe, b_cwe)
                loss     = loss_sem + loss_str + loss_cwe

                epoch_loss_sem += loss_sem.item()
                epoch_loss_str += loss_str.item()
                epoch_loss_cwe += loss_cwe.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                

                epoch_loss  += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            scheduler.step(avg_loss)
            loss_history.append(avg_loss)
            lr_history.append(optimizer.param_groups[0]["lr"])

            loss_sem_history.append(epoch_loss_sem / num_batches)
            loss_str_history.append(epoch_loss_str / num_batches)
            loss_cwe_history.append(epoch_loss_cwe / num_batches)

            if epoch % 25 == 0 or epoch == 1:
                log.info(
                    f"Epoch {epoch:>4}/{epochs} "
                    f"loss={avg_loss:.6f} "
                    f"sem={loss_sem_history[-1]:.6f} "
                    f"str={loss_str_history[-1]:.6f} "
                    f"cwe={loss_cwe_history[-1]:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if avg_loss < best_loss - 1e-6:
                best_loss    = avg_loss
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
                best_epoch = epoch
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    log.info(
                        f"Early stopping at epoch {epoch}  "
                        f"(best_loss={best_loss:.6f})"
                    )
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            log.info(f"Restored best checkpoint (loss={best_loss:.6f})")

        log.info("TripleProjectionFusionAE training complete")
        return model,loss_history,lr_history,best_epoch,loss_sem_history,loss_str_history,loss_cwe_history

    # -----------------------------------------------------------------------
    # STEP 5 — Extract features
    # -----------------------------------------------------------------------
    def _extract_node_features(
        self,
        model: TripleProjectionFusionAE,
        sem:   np.ndarray,
        str_:  np.ndarray,
        cwe:   np.ndarray,
    ):
        """
        Returns:
            node_features : np.ndarray [N, 256]
                Fused bottleneck — fed to DOMINANT and CoLA.

            raw_features  : np.ndarray [N, 841]
                [sem(768) | str(9) | cwe(64)]
                Same three modalities, same information, un-fused.
                Fed to AnomalyDAE as the controlled comparison.
                If DOMINANT/CoLA outperform AnomalyDAE on synthetic
                anomalies, the triple-projection fusion added value.

            diagnostics   : np.ndarray [N, 10]
                [str_error(9) | mse_sem(1)]
                Not a GNN input. Used after scoring to explain flagged
                CVEs: which CVSS metric failed reconstruction and how
                hard the semantic side was to preserve.
        """
        log.info("─" * 60)
        log.info("STEP 5 — Extracting node features (inference pass)")
        log.info("─" * 60)

        model.eval()
        x_sem = torch.from_numpy(sem).to(self.device_torch)
        x_str = torch.from_numpy(str_).to(self.device_torch)
        x_cwe = torch.from_numpy(cwe).to(self.device_torch)

        with torch.no_grad():
            fused, recon_sem, recon_str, recon_cwe = model(x_sem, x_str, x_cwe)

            # ── Interpretability diagnostics (not GNN input) ─────────────
            str_error = torch.abs(recon_str - x_str)              # [N, 9]
            mse_sem   = (
                F.mse_loss(recon_sem, x_sem, reduction="none")
                .mean(dim=1, keepdim=True)
            )                                                       # [N, 1]
            diagnostics = torch.cat([str_error, mse_sem], dim=1)  # [N, 10]

            # ── GNN feature for DOMINANT and CoLA ────────────────────────
            node_features = fused                                   # [N, 256]

            # ── Raw features for AnomalyDAE control ──────────────────────
            raw_features = torch.cat([x_sem, x_str, x_cwe], dim=1) # [N, 841]

        nf   = node_features.cpu().numpy()
        rf   = raw_features.cpu().numpy()
        diag = diagnostics.cpu().numpy()

        log.info(f"Node features (DOMINANT/CoLA)    : {nf.shape}")
        log.info(f"Raw features  (AnomalyDAE)       : {rf.shape}")
        log.info(f"Diagnostics   (interpretability) : {diag.shape}")
        return nf, rf, diag

    # -----------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # -----------------------------------------------------------------------
    def run(
        self,
        epochs:     int   = 500,
        lr:         float = 1e-3,
        batch_size: int   = 1024,
        patience:   int   = 30,
    ) -> dict:
        log.info("=" * 60)
        log.info("APPROACH 2 v4 — TRIPLE-PROJECTION FUSION AUTOENCODER")
        log.info("=" * 60)

        # ── Extract three modalities ──────────────────────────────────────
        sem            = self._get_semantic_embeddings()
        str_           = self._get_structural_features()
        cwe, svd, mlb  = self._get_cwe_features()

        # ── Train ─────────────────────────────────────────────────────────
        model,loss_history,lr_history,best_epoch,loss_sem_h,loss_str_h,loss_cwe_h = self._train_autoencoder(
            sem, str_, cwe,
            epochs=epochs, lr=lr,
            batch_size=batch_size, patience=patience,
        )

        # ── Extract features ──────────────────────────────────────────────
        node_features, raw_features, diagnostics = self._extract_node_features(
            model, sem, str_, cwe
        )

        # ── Save all outputs ──────────────────────────────────────────────
        log.info("─" * 60)
        log.info("STEP 6 — Saving outputs (v4)")
        log.info("─" * 60)

        paths = {
            "node_features":         "node_features_v4.pt",
            "node_features_anomdae": "node_features_anomalydae_v4.pt",
            "diagnostics":           "reconstruction_diagnostics_v4.pt",
            "model_weights":         "triple_projection_model_v4.pt",
            "cve_id_mapping":        "cve_id_mapping_v4.csv",
            "cwe_svd_components":    "cwe_svd_components_v4.npy",
            "cwe_classes":           "cwe_classes_v4.csv",
        }

        torch.save(
            torch.tensor(node_features, dtype=torch.float32),
            paths["node_features"],
        )
        torch.save(
            torch.tensor(raw_features, dtype=torch.float32),
            paths["node_features_anomdae"],
        )
        torch.save(
            torch.tensor(diagnostics, dtype=torch.float32),
            paths["diagnostics"],
        )
        torch.save(model.state_dict(), paths["model_weights"])

        mapping_df = self.df[["CVE_ID"]].copy()
        mapping_df["row_index"] = mapping_df.index
        mapping_df.to_csv(paths["cve_id_mapping"], index=False)

        # SVD components and CWE label index — required for post-scoring
        # interpretability: project flagged CVEs back to named CWE identifiers
        np.save(paths["cwe_svd_components"], svd.components_)
        pd.Series(mlb.classes_).to_csv(
            paths["cwe_classes"], index=False, header=["CWE_ID"]
        )

        # Save training history 
        np.save("loss_history_v4.npy", np.array(loss_history))
        np.save("lr_history_v4.npy",   np.array(lr_history))
        np.save("loss_sem_history_v4.npy", np.array(loss_sem_h))
        np.save("loss_str_history_v4.npy", np.array(loss_str_h))
        np.save("loss_cwe_history_v4.npy", np.array(loss_cwe_h))

        paths["loss_history"] = "loss_history_v4.npy"
        paths["lr_history"]   = "lr_history_v4.npy"
        paths['loss_sem_history'] = "loss_sem_history_v4.npy"
        paths['loss_str_history'] = "loss_str_history_v4.npy"
        paths['loss_cwe_history'] = "loss_cwe_history_v4.npy"

        # ── Summary ───────────────────────────────────────────────────────
        log.info("=" * 60)
        log.info("APPROACH 2 v4 COMPLETE")
        log.info("=" * 60)
        log.info(
            f"  node_features_v4.pt              shape={node_features.shape}"
            f"  [fused(256)] → DOMINANT, CoLA"
        )
        log.info(
            f"  node_features_anomalydae_v4.pt   shape={raw_features.shape}"
            f"  [sem(768)|str(9)|cwe(64)] → AnomalyDAE control"
        )
        log.info(
            f"  reconstruction_diagnostics_v4.pt shape={diagnostics.shape}"
            f"  [str_error(9)|mse_sem(1)] → interpretability only"
        )
        log.info(f"  triple_projection_model_v4.pt    saved")
        log.info(f"  cve_id_mapping_v4.csv            {len(mapping_df):,} rows")
        log.info(
            f"  cwe_svd_components_v4.npy        shape={svd.components_.shape}"
        )
        log.info(f"  cwe_classes_v4.csv               {len(mlb.classes_)} real CWEs")

        log.info(f"loss_history_v4.npy")
        log.info(f"loss_lr_v4.npy")
        log.info(f"loss_sem_history_v4.npy")
        log.info(f"loss_str_history_v4.npy")
        log.info(f"loss_cwe_history_v4.npy")

        log.info(f"Training History Stored")

        plot_training_loss(loss_history, lr_history, best_epoch,save_path="fig_training_loss_v4.pdf")

        return {
            "node_feature_dim":       node_features.shape[1],
            "anomalydae_feature_dim": raw_features.shape[1],
            "num_cves":               len(mapping_df),
            "paths":                  paths,
        }


def plot_training_loss(loss_history, lr_history, best_epoch, save_path="fig_training_loss.pdf"):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss curve
    epochs = list(range(1, len(loss_history) + 1))
    ax1.plot(epochs, loss_history, color="#2563EB", linewidth=1.5, label="Training Loss")
    ax1.axvline(x=best_epoch, color="#DC2626", linestyle="--", linewidth=1.2,
                label=f"Best checkpoint (epoch {best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12, color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")

    # LR on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, lr_history, color="#F59E0B", linewidth=1.0,
             linestyle=":", label="Learning Rate")
    ax2.set_ylabel("Learning Rate", fontsize=12, color="#F59E0B")
    ax2.tick_params(axis="y", labelcolor="#F59E0B")
    ax2.set_yscale("log")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title(
        "Triple-Projection Fusion Autoencoder — Training Convergence",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# ===========================================================================
# MAIN
# ===========================================================================

def main():
    extractor = TripleProjectionFusionExtractor(
        nodes_path     = "nodes_cve.csv",
        cwe_edges_path = "edges_cve_cwe.csv",
        model_name     = "cisco-ai/SecureBERT2.0-biencoder",
    )

    results = extractor.run(
        epochs     = 500,
        lr         = 1e-3,
        batch_size = 1024,
        patience   = 30,
    )

    print()
    print("=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    print(f"  GNN input dim  (DOMINANT / CoLA) : {results['node_feature_dim']}")
    print(f"  GNN input dim  (AnomalyDAE)      : {results['anomalydae_feature_dim']}")
    print(f"  Total CVEs processed             : {results['num_cves']:,}")
    print()
    print("  Files saved:")
    for label, path in results["paths"].items():
        print(f"    {path}")
    print()
    print("  Feed node_features_v4.pt            → DOMINANT, CoLA")
    print("  Feed node_features_anomalydae_v4.pt → AnomalyDAE (control)")
    print("  reconstruction_diagnostics_v4.pt    → post-scoring only")
    print("  cwe_svd_components_v4.npy           → CWE interpretability")
    print("=" * 60)



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error(f"Pipeline failed: {exc}")
        raise