Install cuda-toolkit

```bash
sudo apt install -y nvidia-cuda-toolkit
```

Install mkl

```bash
sudo apt install intel-mkl
```

https://stackoverflow.com/questions/58666921/how-to-set-mkl-on-linux-with-cmake

Prerequisites for First-Time Users
1. To add APT repository access, install the prerequisites:
```bash
sudo apt update
sudo apt install -y gpg-agent wget
```

2. Set up the repository. To do this, download the key to the system keyring:
```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
```

3. Add the signed entry to APT sources and configure the APT client to use the Intel repository:
```bash
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
```

4. Update the packages list and repository index.
```bash
sudo apt update
```

Install with APT
For running applications that require oneMKL:

```bash
sudo apt install intel-oneapi-mkl
```
For developing and compiling oneMKL applications:

```bash
sudo apt install intel-oneapi-mkl-devel
```

Install GoogleTest

```bash
sudo apt update
sudo apt install -y libgtest-dev
```

export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH

Install glog
```bash
sudo apt-get install libgoogle-glog-dev
```