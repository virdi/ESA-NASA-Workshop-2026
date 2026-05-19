# KServe + vLLM InferenceService (Hands‑On)

## What this hands‑on demonstrates

- Using **KServe** to declare a model endpoint via the **InferenceService**
  abstraction. In "raw/standard" mode KServe primarily creates standard
  Kubernetes resources like Deployments and Services (and, depending on your
  cluster integration, an Ingress/Gateway/Route layer).
- Running **vLLM** as the serving runtime for a geospatial model (Prithvi) with
  a TerraTorch IO processor plugin.
- Autoscaling with **KEDA** based on **Prometheus** metrics (external/custom
  metric triggers).

## Disclaimer

These files were tested on:

- **OpenShift 4.20**
- **Red Hat OpenShift AI 3.3.2** (model serving platform based on KServe
  v0.15.2)
- **Red Hat Custom Metrics Autoscaler 2.18.1-2** (uses KEDA 2.18.1)
- **cert-manager Operator for Red Hat OpenShift 1.19.0**

_Some adjustments may be needed for these to work on vanilla Kubernetes or other
OpenShift versions._

> [!WARNING]
>
> From KServe v0.16 onwards, certain fields in the InferenceService
> specification have been renamed, so to use these manifests with newer versions
> of KServe, you will need to update the fields accordingly. For example, the
> `RawDeployment` mode has been renamed to `Standard` mode; the `autoScaling`
> section has had its casing changed to `autoscaling` and the
> `authenticationRef` section has been reworked.

## Repository contents

```text
.
├── Dockerfile
├── kubernetes/
│   ├── prithvi-model-cache-pvc.yaml
│   ├── prithvi-sen1floods11-inferenceservice.yaml
│   ├── prithvi-sen1floods11-podmonitor.yaml
│   ├── prithvi-sen1floods11-predictor-route.yaml
│   ├── prithvi-thanos-reader-clusterrole.yaml
│   ├── prithvi-thanos-reader-clusterrolebinding.yaml
│   ├── prithvi-thanos-reader-sa.yaml
│   ├── prithvi-thanos-reader-sa-token.yaml
│   └── prithvi-thanos-reader-triggerauth.yaml
└── README.md
```

---

## Requirements

### Cluster & tooling

- A working **OpenShift/Kubernetes** cluster and `kubectl` access (or `oc` on
  OpenShift).
- **KServe** installed (this repo uses the `InferenceService` CRD).

### Autoscaling & metrics (required for the YAML as-is)

The provided YAML is configured to scale using KEDA with external metrics. To
run it _without modifications_ you need:

- **Custom Metrics Autoscaler / KEDA** installed and configured.
- **Prometheus** installed, configured and reachable from KServe/KEDA.
- **NVIDIA GPU Operator** installed and configured, and GPU resources
- **GPU Resources**, so that our inference pods can be scheduled.

If you don’t have KEDA + Prometheus/Thanos, you can still deploy by disabling
autoscaling (see **Customization**).

## [OPTIONAL] Build & publish the container image

The `InferenceService` references a container image
(`quay.io/alessandropomponio_ibm/vllm:vllm-v0.18.0-tt-v1.2.5`) that runs
**vLLM** and includes the TerraTorch IO processor plugin. The container image is
built from the `Dockerfile` in this repo and is public, so you can use it
directly. If you want to build and push your own container image, you can do so
using the `Dockerfile` in this repo.

### Prerequisites

- Docker installed and working locally
- A Quay account and a repository you can push to (e.g. `quay.io/<org>/<repo>`)

### 1) Log in to Quay

```bash
docker login quay.io
```

### 2) Build the image

From the repo root (where `Dockerfile` lives):

```bash
# Replace <org> and <repo> with your Quay account and repo
export IMAGE=quay.io/<org>/<repo>:vllm-v0.18.0-tt-v1.2.5

# Build
docker build -t "$IMAGE" .
```

### 3) Push the image

```bash
docker push "$IMAGE"
```

### 4) Point the YAML at your image

Edit `kubernetes/prithvi-sen1floods11-inferenceservice.yaml` and replace the
`image:` value under the predictor container.

---

## Quick start

### 1) Create the namespace

First, create the `prithvi-kserve` namespace:

```bash
kubectl create namespace prithvi-kserve
```

### 2) Apply storage, metrics, networking, and RBAC prerequisites

Apply the PVC, monitoring, RBAC, and (OpenShift) route manifests before creating
the InferenceService:

<details>
<summary>What do these files do?</summary>

- **[`prithvi-model-cache-pvc.yaml`](kubernetes/prithvi-model-cache-pvc.yaml)**:
  Creates a PersistentVolumeClaim with 100Gi storage and ReadWriteMany access
  mode for caching the model files across pods
- **[`prithvi-sen1floods11-podmonitor.yaml`](kubernetes/prithvi-sen1floods11-podmonitor.yaml)**:
  Configures Prometheus monitoring for the InferenceService pods, scraping
  metrics from port 8000 every 30 seconds
- **[`prithvi-thanos-reader-sa.yaml`](kubernetes/prithvi-thanos-reader-sa.yaml)**:
  Creates the `thanos-reader` ServiceAccount used for accessing
  Prometheus/Thanos metrics
- **[`prithvi-thanos-reader-sa-token.yaml`](kubernetes/prithvi-thanos-reader-sa-token.yaml)**:
  Creates a long-lived service account token Secret for the `thanos-reader`
  ServiceAccount
- **[`prithvi-thanos-reader-clusterrole.yaml`](kubernetes/prithvi-thanos-reader-clusterrole.yaml)**:
  Defines cluster-wide permissions for reading Prometheus metrics, pods, and
  namespaces. This supplements KServe's default permissions by adding the
  missing `prometheuses/api` GET permission required for metrics access
- **[`prithvi-thanos-reader-clusterrolebinding.yaml`](kubernetes/prithvi-thanos-reader-clusterrolebinding.yaml)**:
  Binds the ClusterRole to the `thanos-reader` ServiceAccount
- **[`prithvi-thanos-reader-triggerauth.yaml`](kubernetes/prithvi-thanos-reader-triggerauth.yaml)**:
  Configures KEDA authentication to access Thanos/Prometheus metrics for
  autoscaling
- **[`prithvi-sen1floods11-predictor-route.yaml`](kubernetes/prithvi-sen1floods11-predictor-route.yaml)**:
  Creates an OpenShift Route with TLS termination for external access to the
  InferenceService. Includes annotations to ensure traffic is routed in round
  robin across InferenceService pods with no cookie-based session affinity.

</details>

```bash
kubectl apply -f kubernetes/prithvi-model-cache-pvc.yaml
kubectl apply -f kubernetes/prithvi-sen1floods11-podmonitor.yaml
kubectl apply -f kubernetes/prithvi-thanos-reader-sa.yaml
kubectl apply -f kubernetes/prithvi-thanos-reader-sa-token.yaml
kubectl apply -f kubernetes/prithvi-thanos-reader-clusterrole.yaml
kubectl apply -f kubernetes/prithvi-thanos-reader-clusterrolebinding.yaml
kubectl apply -f kubernetes/prithvi-thanos-reader-triggerauth.yaml

# OpenShift only:
oc apply -f kubernetes/prithvi-sen1floods11-predictor-route.yaml
```

### 3) Deploy the InferenceService

```bash
kubectl apply -f kubernetes/prithvi-sen1floods11-inferenceservice.yaml
```

> [!NOTE]
>
> The KEDA autoscaling configuration in this setup is designed to scrape metrics
> from the default Thanos instance that comes bundled with OpenShift. If you're
> using a different Kubernetes distribution or custom monitoring setup, you may
> need to adjust the KEDA trigger authentication accordingly.

### 4) Wait for rollout

```bash
kubectl wait --for=condition=ready pod -l serving.kserve.io/inferenceservice=prithvi-sen1floods11 --timeout=300s
```

You can also check the InferenceService status:

```bash
kubectl get inferenceservice prithvi-sen1floods11
kubectl describe inferenceservice prithvi-sen1floods11
```

### 5) [OPTIONAL] Check logs

```bash
kubectl get pods
kubectl logs -l serving.kserve.io/inferenceservice=prithvi-sen1floods11 --tail=200
```

### 6) [OPTIONAL] Check the network endpoints

How you reach the service depends on how networking is configured in your
cluster (Ingress / Gateway / Route).

Common checks:

```bash
# Look for a URL in the InferenceService status (if populated)
kubectl get inferenceservice prithvi-sen1floods11 -o jsonpath='{.status.address.url}'

# Discover created networking objects (varies by distro)
kubectl get svc,ingress,httproute,route | grep prithvi-sen1floods11
```

---

## What's inside `kubernetes/prithvi-sen1floods11-inferenceservice.yaml` (high level)

The manifest defines a KServe InferenceService configured for “raw/standard”
deployment with KEDA autoscaling.

It also includes:

- Prometheus scrape annotations for `/metrics`
- A predictor container that runs `vllm serve` with model/plugin args
- Resource requests/limits including `nvidia.com/gpu: "1"`
- `minReplicas` / `maxReplicas`
- An external autoscaling metric driven by a Prometheus query

### Model & runtime details

The example serves:

- Model: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`
- vLLM args include:
  - `--skip-tokenizer-init`
  - `--enable-mm-embeds`
  - `--io-processor-plugin terratorch_segmentation`
  - `--max-num-seqs 1`
  - `--enforce-eager`

Container image used in examples:

- `quay.io/alessandropomponio_ibm/vllm:vllm-v0.18.0-tt-v1.2.5`

---

## Scaling demo (KServe + KEDA)

Before generating load, confirm:

- Prometheus is scraping vLLM pods via the PodMonitor
- KEDA authentication objects are in place
- (OpenShift) any Route/load-balancing config you rely on is present

Get the base URL for your InferenceService:

```bash
# For OpenShift (using Route):
export BASE_URL="https://$(oc get route prithvi-sen1floods11-predictor-roundrobin -o jsonpath='{.spec.host}')"

# For standard Kubernetes (using port-forward):
kubectl port-forward svc/prithvi-sen1floods11-predictor 8080:80 &
export BASE_URL="http://localhost:8080"
```

Generate load using `vllm bench`:

```bash
vllm bench serve \
  --base-url "$BASE_URL" \
  --dataset-name=custom \
  --model ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 \
  --seed 12345 \
  --skip-tokenizer-init \
  --endpoint /pooling \
  --backend vllm-pooling \
  --metric-percentiles 25,75,99 \
  --percentile-metrics e2el \
  --dataset-path ./dataset_url_input_india.jsonl \
  --num-prompts 7000 \
  --max-concurrency 100 \
  --ramp-up-strategy linear \
  --ramp-up-start-rps 10 \
  --ramp-up-end-rps 70 \
  --insecure
```

Watch replicas change:

```bash
watch kubectl get pods
```

Watch the metrics that KEDA sees:

```bash
watch kubectl get hpa
```

---

## Customization

### Reduce resource requests

If your cluster is smaller, edit `resources.requests` / `resources.limits` to
values that your nodes can satisfy.

### Disable autoscaling (if you don’t have KEDA/Prometheus)

To run without KEDA + Prometheus/Thanos:

- Remove `minReplicas` and `maxReplicas`
- Remove the `autoScaling:` section entirely
- Remove the `serving.kserve.io/autoscalerClass: "keda"` annotation (if present)

You can also skip applying:

- `kubernetes/prithvi-sen1floods11-podmonitor.yaml`
- `kubernetes/prithvi-thanos-reader-clusterrole.yaml`
- `kubernetes/prithvi-thanos-reader-clusterrolebinding.yaml`
- `kubernetes/prithvi-thanos-reader-triggerauth.yaml`

And if you are not exposing through OpenShift routing, you can skip:

- `kubernetes/prithvi-sen1floods11-predictor-route.yaml`

---

## Troubleshooting

### Pods stuck in Pending

Common causes: insufficient CPU/RAM, no GPU node, or missing NVIDIA device
plugin.

```bash
kubectl describe pod <pod-name>
```

### Container starts but model fails to load

```bash
kubectl logs <pod-name> --tail=300
```

### Endpoint not reachable

Verify how your cluster exposes KServe services
(Ingress/Route/Gateway/HTTPRoute) and confirm the required networking
integration is installed and configured.
