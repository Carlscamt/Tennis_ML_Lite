# System Architecture

## Component Map

```mermaid
graph TD
    %% Styling
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef logic fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef data fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef infra fill:#eceff1,stroke:#455a64,stroke-width:2px,stroke-dasharray: 5 5;

    %% --- INFRASTRUCTURE ---
    subgraph Infra ["Infrastructure"]
        Docker["Docker Container<br/>(python:3.11-slim)"]:::infra
    end

    %% --- INTERFACE ---
    subgraph CLI ["CLI Interface"]
        TennisCLI["tennis.py<br/>(Entry Point)"]:::ui
        CmdTrain["train"]:::ui
        CmdPredict["predict"]:::ui
        CmdAudit["audit"]:::ui
        CmdServe["serving-config"]:::ui
    end

    %% --- BACKEND ORCHESTRATION ---
    subgraph Backend ["Backend Core (src/)"]
        Pipeline["src/pipeline.py<br/>(TennisPipeline Class)"]:::logic
        DataPipe["run_data_pipeline()<br/>(ETL)"]:::logic
        TrainPipe["run_training_pipeline()<br/>(Training)"]:::logic
    end

    %% --- SERVICES ---
    subgraph Services ["Services"]
        Scraper["src/scraper.py<br/>(Data Fetching)"]:::logic
        FeatEng["src/transform/<br/>(Feature Engineering)"]:::logic
        
        subgraph ModelOps ["Model Operations"]
            registry["src/model/registry.py<br/>(ModelRegistry)"]:::model
            server["src/model/serving.py<br/>(ModelServer)"]:::model
        end
        
        obs["src/utils/observability.py<br/>(Logs/Metrics)"]:::infra
    end

    %% --- DATA ---
    subgraph Storage ["Data Layer"]
        RawData[("data/raw/*.parquet")]:::data
        ProcData[("data/processed/features.parquet")]:::data
        
        subgraph RegistryStore ["models/registry/"]
            ProdModel[("production/v1.2.0/")]:::data
            StageModel[("staging/v1.3.0/")]:::data
            ExpModel[("experiments/")]:::data
        end
        
        Outputs[("Reporting<br/>CSV/JSON/Parquet")]:::data
    end

    %% --- FLOWS ---
    Docker -- "Hosts" --> CLI
    
    TennisCLI --> CmdPredict
    TennisCLI --> CmdTrain
    TennisCLI --> CmdAudit
    TennisCLI --> CmdServe
    
    CmdPredict --> Pipeline
    CmdTrain --> TrainPipe
    
    %% Internal Flows
    Pipeline --> |"Calls"| Scraper
    Pipeline --> |"Calls"| DataPipe
    Pipeline --> |"Predict Batch"| server
    Pipeline --> |"Logs to"| obs
    
    %% Model Serving Logic
    server --> |"Load Champion"| registry
    server --> |"Load Challenger"| registry
    server --> |"Shadow/Canary"| server
    
    registry --> |"Read/Write"| RegistryStore
    
    DataPipe --> |"Reads"| RawData
    DataPipe --> |"Uses"| FeatEng
    DataPipe --> |"Writes"| ProcData
    
    TrainPipe --> |"Reads"| ProcData
    TrainPipe --> |"Register"| registry
    
    Scraper --> |"Write"| RawData
    
    TennisCLI --> |"--output"| Outputs
```

## Module Responsibilities

### CLI & Infrastructure
- **`tennis.py`**: Command-line entry point using `argparse`. Dispatches subcommands (`train`, `predict`, `audit`, `promote`).
- **`Dockerfile`**: Defines the reproducible runtime environment.
- **`run_daily.bat`**: Windows automation script for scheduled execution.

### Backend Core (`src/pipeline.py`)
- **`TennisPipeline`**: High-level orchestration class.
  - Integrates **Observability** context managers for tracing.
  - Manages **Data Quality** gates (Schema validation, drift detection).
- **`run_data_pipeline`**: Orchestrates ETL (Raw -> Processed).

### Services

#### Model Operations (`src.model`)
- **`src.model.registry`**:
    - **Artifact Management**: Stores models in `models/registry/{stage}/{version}/`.
    - **Stages**: `experimental` -> `staging` -> `production` -> `archived`.
    - **Metadata**: Tracks metrics, timestamps, and promotion history in `model.meta.json`.
- **`src.model.serving`**:
    - **Advanced Serving**: Implements Canary deployments (percentage traffic), Shadow Mode (silent parallel execution), and Fallback (reliability).
    - **Loading**: Uniform loading interface for `xgboost` (native) and `joblib` (sklearn/pipeline) artifacts.
    - **Observability**: detailed logs for model selection and latency.

#### Data & Utilities
- **`src.scraper`**: Data collection with rate limiting, circuit breakers, and raw parquet storage.
- **`src.transform`**: Feature engineering logic and Pandera schema validation.
- **`src.utils.observability`**: Centralized structured logging (`structlog`) and Prometheus metrics integration.

### CI/CD
- **`.github/workflows/ci.yaml`**: Automated testing workflow.
    - Runs Unit Tests (`pytest tests/unit`).
    - Runs Integration Tests (`pytest tests/integration`).
    - Validates Code Quality (Linting).

## Serving Logic Flow

The **ModelServer** implements a sophisticated routing engine for inference:

1.  **Initialization**: Loads `active_model` (Production) and optional `challenger_model` (Staging).
2.  **Request Handling**:
    *   **Shadow Mode**: If enabled, predicts with Champion (returned to user) AND Challenger (logged for comparison).
    *   **Canary Mode**: If within canary percentage, predicts with Challenger.
    *   **Fallback**: If Champion fails prediction, automatically falls back to Challenger to prevent outage.
3.  **Response**: Returns structured JSON with prediction, probabilities, model version used, and serving mode (`champion_only`, `canary`, `shadow`, `fallback`).
