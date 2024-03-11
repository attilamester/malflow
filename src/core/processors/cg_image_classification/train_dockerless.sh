log() {
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [Train dockerless] $1"
}

if [ $# -ne 1 ]; then
    log "Usage: $0 <path-to-tb-log-dir>"
    exit 1
fi

./download_dependencies.sh main.py

(
  cd ../../../ && \
  python3 -m core.processors.cg_image_classification.main \
    -m core.processors.cg_image_classification.train_definitions \
    -tb $1 \
    --checkpoints $1 \
    --print-freq 100 \
    "${1:-''}"
)
