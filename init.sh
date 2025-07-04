#!/bin/sh

# Default trainer
TRAINER="aitoolkit"

# Check if any arguments are passed
if [ $# -eq 0 ]; then
    # No arguments, use default behavior
    INIT_URL="https://geocine.github.io/flux/init/${TRAINER}.sh"
    curl -s "$INIT_URL" | sh
else
    # Arguments passed, use bash for more advanced parsing
    exec bash -s -- "$@" << 'EOF'
#!/bin/bash

TRAINER="aitoolkit"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --trainer)
        TRAINER="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
done

# Validate trainer
valid_trainers=("simpletuner" "trainer" "aitoolkit", "atchroma")
if [[ ! " ${valid_trainers[@]} " =~ " ${TRAINER} " ]]; then
    echo "Invalid trainer specified. Valid options are: ${valid_trainers[*]}"
    exit 1
fi

# URL for the specific trainer's init script
INIT_URL="https://geocine.github.io/flux/init/${TRAINER}.sh"

curl -s "$INIT_URL" | sh

EOF
fi