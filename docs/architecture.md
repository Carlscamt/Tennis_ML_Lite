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
        registry["src/model/registry.py<br/>(Model Registry)"]:::model
        server["src/model/serving.py<br/>(Batch Server)"]:::model
        obs["src/utils/observability.py<br/>(Logs/Metrics)"]:::infra
    end

    %% --- DATA ---
    subgraph Storage ["Data Layer"]
        RawData[("data/raw/*.parquet")]:::data
        ProcData[("data/processed/features.parquet")]:::data
        ModelFile[("models/registry/*.json")]:::data
        Outputs[("Reporting<br/>CSV/JSON/Parquet")]:::data
    end

    %% --- FLOWS ---
    Docker -- "Hosts" --> CLI
    
    TennisCLI --> |"predict"| Pipeline
    TennisCLI --> |"scrape"| Scraper
    TennisCLI --> |"train"| TrainPipe
    
    %% Internal Flows
    Pipeline --> |"Calls"| Scraper
    Pipeline --> |"Calls"| DataPipe
    Pipeline --> |"Uses"| server
    Pipeline --> |"Logs to"| obs
    
    server --> |"Load Best"| registry
    registry --> |"Read"| ModelFile

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
- **`tennis.py`**: Command-line entry point. Handles argument parsing and dispatching.
- **`Dockerfile`**: Defines the reproducible runtime environment.
- **`run_daily.bat`**: Windows automation script for scheduled execution.

### Backend Core (`src/pipeline.py`)
- **`TennisPipeline`**: High-level orchestration class.
  - Integrates **Observability** context managers for tracing.
  - Manages **Data Quality** gates (Schema validation, drift detection).
- **`run_data_pipeline`**: Orchestrates ETL (Raw -> Processed).
- **`run_training_pipeline`**: Orchestrates Training (Processed -> Model Registry).

### Services
- **`src.scraper`**: Data collection with rate limiting and circuit breakers.
- **`src.transform`**: Feature engineering logic.
- **`src.model.registry`**: Manages model versions and promotion stages (Experimental -> Production).
- **`src.model.serving`**: Handles batch prediction requests using the best available model.
- **`src.utils.observability`**: Centralized structured logging (`structlog`) and Prometheus metrics.

### CI/CD
- **`.github/workflows/ci.yaml`**: Automated testing workflow triggered on push.
