import os
import hydra
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def configure_logger(output_dir: str) -> str:
    log_path = os.path.join(output_dir, "act_separation.log")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return log_path


def load_model(model_name: str):
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def extract_activations(code: str, tokenizer, model) -> np.ndarray:
    """
    Returns array of shape (num_layers, hidden_size)
    One vector per transformer layer for the final token.
    """
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.hidden_states: tuple of (num_layers+1) tensors
    # each tensor shape: (batch=1, seq_len, hidden_size)
    # We take the last token position: index [-1]
    hidden_states = outputs.hidden_states  # includes embedding layer
    # Stack all layers: shape → (num_layers+1, hidden_size)
    activations = torch.stack([
        layer[0, -1, :]   # last token of layer
        for layer in hidden_states
    ]).numpy()             # (num_layers+1, hidden_size)
    return activations


def collect_all_activations(examples, labels, tokenizer, model):
    """
    Returns:
        all_activations: (num_examples, num_layers, hidden_size)
        all_labels:      (num_examples,)
    """
    all_activations = []
    for code in examples:
        acts = extract_activations(code, tokenizer, model)
        all_activations.append(acts)
    return np.array(all_activations), np.array(labels)


def run_linear_probe(all_activations: np.ndarray, labels: np.ndarray):
    """
    Trains a logistic regression probe at each layer.
    Returns accuracy per layer.

    all_activations: (n_examples, n_layers, hidden_size)
    labels:          (n_examples,)
    """
    n_examples, n_layers, hidden_size = all_activations.shape
    logger.info(
        f"Running linear probe: {n_examples} examples, {n_layers} layers, hidden_size={hidden_size}"
    )
    accuracies = []
    for layer_idx in range(n_layers):
        X = all_activations[:, layer_idx, :]   # (n_examples, hidden_size)
        # Standardise features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Logistic regression — linear probe
        probe = LogisticRegression(max_iter=1000, C=1.0)
        # Cross-val accuracy (leave-one-out for small datasets)
        if n_examples >= 5:
            cv = min(5, n_examples) # 5-fold or less if very small dataset
            scores = cross_val_score(probe, X_scaled, labels, cv=cv)
            acc = scores.mean()
        else:
            probe.fit(X_scaled, labels)
            acc = probe.score(X_scaled, labels)
        accuracies.append(acc)
        logger.info(f"Layer {layer_idx:2d}: accuracy = {acc:.3f}")
    return accuracies


def compute_steering_vector(
    secure_acts: np.ndarray,
    insecure_acts: np.ndarray,
    layer_idx: int
) -> np.ndarray:
    """
    Δ_security = mean(secure activations) - mean(insecure activations)
    at a specific layer.

    Shape: (hidden_size,)
    """
    mu_secure   = secure_acts[:, layer_idx, :].mean(axis=0)
    mu_insecure = insecure_acts[:, layer_idx, :].mean(axis=0)
    delta = mu_secure - mu_insecure

    cosine_sim = np.dot(mu_secure, mu_insecure) / (
        np.linalg.norm(mu_secure) * np.linalg.norm(mu_insecure) + 1e-10
    )
    logger.info(f"Steering vector at layer {layer_idx}:")
    logger.info(f"||Δ|| = {np.linalg.norm(delta):.4f}")
    logger.info(f"cosine(μ_secure, μ_insecure) = {cosine_sim:.4f}")
    logger.info("(lower cosine = more separated = better)")
    return delta


def visualise_separation(
    all_activations: np.ndarray,
    labels: np.ndarray,
    layer_idx: int,
    save_path: str
):
    X = all_activations[:, layer_idx, :]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71" if l == 0 else "#e74c3c" for l in labels]
    markers = ["o" if l == 0 else "X" for l in labels]

    for i, (x, y) in enumerate(X_2d):
        ax.scatter(x, y, c=colors[i], marker=markers[i],
                s=120, edgecolors="white", linewidth=0.8, zorder=3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
            markersize=12, label="Secure"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#e74c3c",
               markersize=12, label="Insecure"),
    ]
    ax.legend(handles=legend_elements, fontsize=12)

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(f"Activation Space Separation at Layer {layer_idx}\n"
                f"(Secure vs Insecure Code)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Plot saved to: {save_path}")
    plt.close()


def plot_probe_accuracies(
    accuracies: list,
    save_path: str = "act_edit/output/probe_accuracy_by_layer.png"
):
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(len(accuracies)))

    ax.plot(layers, accuracies, "o-", color="#3498db",
            linewidth=2, markersize=7, zorder=3)
    ax.axhline(0.5, color="#e74c3c", linestyle="--",
            linewidth=1.5, label="Chance (0.5)", zorder=2)

    best_layer = int(np.argmax(accuracies))
    ax.axvline(best_layer, color="#2ecc71", linestyle=":",
            linewidth=1.5, label=f"Best layer: {best_layer}", zorder=2)

    ax.fill_between(layers, 0.5, accuracies,
                    where=[a > 0.5 for a in accuracies],
                    alpha=0.15, color="#3498db", label="Above chance")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy (5-fold CV)", fontsize=12)
    ax.set_title("Linear Probe: Secure vs Insecure Code\n"
                "Accuracy by Layer Depth", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Probe accuracy plot saved to: {save_path}")
    plt.close()



# ──────────────────────────────────────────────
# 8. MAIN PIPELINE
# ──────────────────────────────────────────────
@hydra.main(version_base=None, config_path="../config", config_name="general")
def main(config: DictConfig):
    log_path = configure_logger(config["output"]["dir"])
    logger.info(f"Writing run logs to {log_path}")
    # Load model
    tokenizer, model = load_model(config["model"]["name"])
    n_layers = model.config.num_hidden_layers
    logger.info(f"Model has {n_layers} transformer layers")
    logger.info(f"Hidden size: {model.config.hidden_size}")

    SECURE_EXAMPLES = config["data"]["secure_examples"]
    INSECURE_EXAMPLES = config["data"]["insecure_examples"]

    if len(SECURE_EXAMPLES) == 0 or len(INSECURE_EXAMPLES) == 0:
        raise ValueError(
            "Config must provide non-empty data.secure_examples and data.insecure_examples"
        )
    
    # Build labelled dataset
    all_code    = SECURE_EXAMPLES + INSECURE_EXAMPLES
    all_labels  = [0] * len(SECURE_EXAMPLES) + [1] * len(INSECURE_EXAMPLES)

    # Extract activations
    logger.info("Extracting activations...")
    all_activations, labels = collect_all_activations(
        all_code, all_labels, tokenizer, model
    )
    logger.info(f"Activation tensor shape: {all_activations.shape}")
    # (n_examples, n_layers+1, hidden_size)

    # Split for steering vector computation
    n_secure = len(SECURE_EXAMPLES)
    secure_acts   = all_activations[:n_secure]
    insecure_acts = all_activations[n_secure:]

    # Run linear probe at each layer
    accuracies = run_linear_probe(all_activations, labels)
    best_layer = int(np.argmax(accuracies))
    logger.info(
        f"Best probe layer: {best_layer} (accuracy={accuracies[best_layer]:.3f})"
    )

    # Compute steering vector at best layer
    delta = compute_steering_vector(secure_acts, insecure_acts, best_layer)

    # Visualise
    plot_probe_accuracies(accuracies, save_path=config["output"]["probe_accuracy_plot"])
    visualise_separation(
        all_activations,
        labels,
        best_layer,
        save_path=config["output"]["activation_separation_plot"],
    )

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Chance baseline:      0.500")
    logger.info(f"Best probe accuracy:  {accuracies[best_layer]:.3f}  (layer {best_layer})")
    logger.info(f"Separation exists:    {'YES' if accuracies[best_layer] > 0.65 else 'WEAK/NO'}")
    logger.info(f"Steering vector norm: {np.linalg.norm(delta):.4f}")
    logger.info("Next step: inject alpha * delta into model activations at")
    logger.info(f"layer {best_layer} during generation and measure vulnerability rate change.")


if __name__ == "__main__":
    main()