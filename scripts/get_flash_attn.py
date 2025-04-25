import platform
import sys
import requests
from bs4 import BeautifulSoup
import subprocess
import torch


def get_cuda_version():
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        return f"cu{cuda_version.replace('.', '')[:2]}"  # 例如：cu121
    return "cpu"


def get_torch_version():
    return f"torch{torch.__version__.split('+')[0]}"[:-2]  # 例如：torch2.2


def get_python_version():
    version = sys.version_info
    return f"cp{version.major}{version.minor}"  # 例如：cp310


def get_abi_flag():
    return "abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "abiFALSE"


def get_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine == "x86_64":
        return "linux_x86_64"
    elif system == "windows" and machine == "amd64":
        return "win_amd64"
    elif system == "darwin" and machine == "x86_64":
        return "macosx_x86_64"
    else:
        raise ValueError(f"Unsupported platform: {system}_{machine}")


def generate_flash_attn_filename(flash_attn_version="2.7.2.post1"):
    cuda_version = get_cuda_version()
    torch_version = get_torch_version()
    python_version = get_python_version()
    abi_flag = get_abi_flag()
    platform_tag = get_platform()

    filename = (
        f"flash_attn-{flash_attn_version}+{cuda_version}{torch_version}cxx11{abi_flag}-"
        f"{python_version}-{python_version}-{platform_tag}.whl"
    )
    return filename

def install_flash_attn_wheel(filename):
    base_url = "https://github.com/Dao-AILab/flash-attention/releases"
    releases_url = f"{base_url}"

    print(f"Searching for {filename} on FlashAttention GitHub releases...")

    # Fetch release page HTML
    response = requests.get(releases_url)
    if response.status_code != 200:
        print("Failed to fetch releases page.")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all("a")

    download_url = None
    for link in links:
        href = link.get("href", "")
        if filename in href:
            download_url = "https://github.com" + href
            break

    if download_url:
        print(f"Found wheel: {download_url}")
        print("Downloading and installing...")

        try:
            subprocess.run(["pip", "install", download_url], check=True)
            print("FlashAttention installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"pip install failed: {e}")
    else:
        print(f"Wheel '{filename}' not found on FlashAttention releases.")

if __name__ == "__main__":
    try:
        filename = generate_flash_attn_filename()
        print("Generated filename:", filename)
        install_flash_attn_wheel(filename)
    except Exception as e:
        print("Error generating filename:", e)