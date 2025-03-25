echo "Setting up the project..."
echo "Detecting operating system..."

DEVICE_TYPE=""

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
git submodule init && git submodule update
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
uv pip install -U jax "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
uv pip install evosax==0.1.4
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
echo "you can get your token from https://huggingface.co/settings/tokens"
echo ""
echo "###############################################"
echo ""