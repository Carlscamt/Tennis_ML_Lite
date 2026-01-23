# System Architecture

## Component Map

```mermaid
graph TD
    %% Styling
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef logic fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef data fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% --- INTERFACE ---
    subgraph CLI ["CLI Interface"]
        TennisCLI["tennis.py<br/>(Entry Point)"]:::ui
    end

    %% --- BACKEND ORCHESTRATION ---
    subgraph Backend ["Backend Core (src/)"]
        Pipeline["src/pipeline.py<br/>(TennisPipeline Class)"]:::logic
        DataPipe["run_data_pipeline()<br/>(src/pipeline.py)"]:::logic
        TrainPipe["run_training_pipeline()<br/>(src/pipeline.py)"]:::logic
    end

    %% --- COMPONENTS ---
    subgraph Services ["Services"]
        Scraper["src/scraper.py<br/>(Data Fetching)"]:::logic
        FeatEng["src/transform/<br/>(Feature Engineering)"]:::logic
        Trainer["src/model/trainer.py<br/>(XGBoost Training)"]:::model
        Predictor["src/model/predictor.py<br/>(Inference)"]:::model
    end

    %% --- DATA ---
    subgraph Storage ["Data Layer"]
        RawData[("data/raw/*.parquet")]:::data
        ProcData[("data/processed/features.parquet")]:::data
        ModelFile[("models/xgboost_model")]:::data
    end

    %% --- FLOWS ---
    
    TennisCLI --> |"predict"| Pipeline
    TennisCLI --> |"scrape"| Scraper
    TennisCLI --> |"train"| TrainPipe

    %% Internal Flows
    Pipeline --> |"Calls"| Scraper
    Pipeline --> |"Calls"| DataPipe
    Pipeline --> |"Uses"| Predictor
    
    DataPipe --> |"Reads"| RawData
    DataPipe --> |"Uses"| FeatEng
    DataPipe --> |"Writes"| ProcData
    
    TrainPipe --> |"Reads"| ProcData
    TrainPipe --> |"Train"| Trainer
    Trainer --> |"Save"| ModelFile
    
    Predictor --> |"Load"| ModelFile
    
    Scraper --> |"Write"| RawData
```

## Module Responsibilities

### CLI
- **`tennis.py`**: Command-line interface. Wraps `src/pipeline.py` functions for scraping, training, and predicting.

### Backend Core (`src/pipeline.py`)
- **`TennisPipeline`**: High-level class. Consumes `Upcoming` data, calls `Predictor`, and applies value bet logic.
- **`run_data_pipeline`**: ETL function. `Raw -> Dedupe -> Features -> Processed`.
- **`run_training_pipeline`**: Model training function. `Processed -> Split -> Train -> Save`.

### Services
- **`src.scraper`**: Handles all scraping (Historical, Upcoming, Active Players). Includes "Smart Update" state management.
- **`src.transform`**: Feature engineering logic (`features.py`, `feature_engineer.py`).
- **`src.model`**: Wrapper around XGBoost/Sklearn for training (`trainer.py`) and inference (`predictor.py`).
