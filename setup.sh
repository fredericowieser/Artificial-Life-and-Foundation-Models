echo "Setting up the project..."
echo "Detecting operating system..."

DEVICE_TYPE=""

# Prevent Hugging Face Tokenizer from throwing warnings
# when using tqdm. This is a workaround for the issue.
# This issue shouldn't break anything, but it's better
# to prevent the warnings from showing up.
export TOKENIZERS_PARALLELISM=false

# Detect OS and assign to DEVICE_TYPE
case "$(uname -s)" in
    "Darwin")
        DEVICE_TYPE="mac"
        ;;
    "Linux")
        if [ -f /etc/debian_version ]; then
            DEVICE_TYPE="debian"
        elif [ -f /etc/arch-release ]; then
            DEVICE_TYPE="arch"
        else
            echo "Unsupported Linux distribution"
            exit 1
        fi
        ;;
    "MINGW"*|"MSYS"*|"CYGWIN"*)
        DEVICE_TYPE="windows"
        ;;
    *)
        echo "Unsupported operating system"
        exit 1
        ;;
esac
echo "Operating system detected: $DEVICE_TYPE"

# Initialize the submodules
echo "Initializing submodules..."
git submodule update --init --recursive
echo "Submodules initialized."

# Install the dependencies using uv
echo "Checking if uv is installed..."
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    case $DEVICE_TYPE in
        "mac"|"debian")
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"  # Ensure uv is found
            ;;
        "arch")
            sudo pacman -S uv
            ;;
        "windows")
            powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
            ;;
    esac
fi

# Explicitly reload PATH to make sure uv is found
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo "uv is installed."

# Create and activate virtual environment
echo "Creating virtual environment..."
uv venv .venv
uv venv --python 3.11

# Install requirements in virtual environment
echo "Installing dependencies..."
uv pip install --upgrade pip
uv pip install -r requirements.txt
uv pip install "huggingface_hub[cli]"
# Check if cuda is available, install jax with cuda support
if nvcc --version &> /dev/null; then
    echo "CUDA is available. Installing JAX with CUDA support..."
    uv pip install -U "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
else
    echo "CUDA is not available."
fi
echo "Virtual environment created."
echo ""
echo "############### NOTE TO USER ##################"
echo "###############################################"
echo ""
echo "To activate the virtual environment, run the following command:"
echo ""
if [ $DEVICE_TYPE = "windows" ]; then
    echo ".\\.venv\\Scripts\\Activate.ps1"
else
    echo "source .venv/bin/activate"
fi
echo ""
echo "To use some of the project you need to give you Hugging Face Token"
echo "You can do this by running the following command:"
echo ""
echo "huggingface-cli login"
echo ""
echo "You can get your token from https://huggingface.co/settings/tokens"
echo ""
echo "###############################################"
echo ""