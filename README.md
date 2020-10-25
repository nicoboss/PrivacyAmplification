# Privacy Amplification v1.1.

## Installation without modification

**Windows:**

Execute [http://www.nicobosshard.ch/PrivacyAmplificationSetupWin64_v1.1.0.exe](http://www.nicobosshard.ch/PrivacyAmplificationSetupWin64_v1.1.0.exe)

On the Desktop there will be 4 Icons: PrivacyAmplification, MatrixSeedServerExample,
SendKeysExample and ReceiveAmpOutExample. Open all the 4 Icons and the program will run. The
config file will be in C:\Program Files\PrivacyAmplification\PrivacyAmplification\config.yaml

**Linux:**

Install NVidia Drivers, Install NVidia Docker, git clone --recursive https://oauth2:_tqToxodj7-
zxRsANcyD@gitlab.enterpriselab.ch/fm-20hs/privacyamplification.git and execute ./run_docker.sh.
After Docker has started you will be in the Container and see everything running.

## How to modify

### Windows:
Open a git Client and enter: git clone --recursive https://oauth2:_tqToxodj7-
zxRsANcyD@gitlab.enterpriselab.ch/fm-20hs/privacyamplification.git

Optionally: Open examples/MatrixSeedServerExample/MatrixSeedServerExample.sln with Visual
Studio 2019 and make it interface with your RNG generator. For this replace
"fromFile("toeplitz_seed.bin", toeplitz_seed);” with a function that takes toeplitz_seed as argument
and fills it with cryptographically secure random numbers.

Open examples/SendKeysExample/SendKeysExample.sln with Visual Studio 2019 and make it
interface with your FPGA board. For this replace "fromFile("keyfile.bin", key_data_alice);” with a
function that takes key_data_alice as argument and fills it with the raw corrected key received and
corrected by your FPGA board

Open examples/ReceiveAmpOutExample/ReceiveAmpOutExample.sln with Visual Studio 2019 and
integrate your final product into it.

To modify PrivacyAmplification open PrivacyAmplification/PrivacyAmplification.sln with Visual Studio
2019.

To edit the config file open and edit config.yaml

To build the Installer you need NSIS. Make sure to build all four projects for release. Right click
PrivacyAmplification.nsi and click "Compile NSIS Script”


### Linux:
Follow "Installation without modification” inside the docker Container switch into the folder that
contains the file you want to edit. Edit it like on Windows. Go back to the main folder and execute
./run.sh to build and run.

Optionally: Open examples/MatrixSeedServerExample/main.cpp and make it interface with your
RNG generator. For this replace "fromFile("toeplitz_seed.bin", toeplitz_seed);” with a function that
takes toeplitz_seed as argument and fills it with cryptographically secure random numbers.

Open examples/SendKeysExample/main.cpp and make it interface with your FPGA board. For this
replace "fromFile("keyfile.bin", key_data_alice);” with a function that takes key_data_alice as
argument and fills it with the raw corrected key received and corrected by your FPGA board

Open examples/ReceiveAmpOutExample/main.cpp and integrate your final product into it.

To modify PrivacyAmplification open PrivacyAmplification/PrivacyAmplification.h and
PrivacyAmplification/PrivacyAmplification.cu

The config.yaml in the build directory is hardlinked with the one in the PrivacyAmplification directory
so either of those can be edited.

**How to use tmux:**\
Linux Docker uses tmux. When starting the container 4 tmux windows will be open. To close them all
detach using "Ctrl & B then D” and enter "./run.sh stop” To close a certain window just enter exit. To
reattach tmux enter "tmux attach”. To switch tmux window use Ctrl & B then arrow key.

**How to stop and exit the docker container:**\
After detaching tmux just enter exit.

**How to start and reattach the docker container:**\
"docker start container_id && docker attach container_id”

**How to SSH into the docker container:**\
A private OpenSSH and PuTTY key should generate in the same folder as run_docker.sh. The IP is the
same as host IP and the Port is 2222.

**How to update the container:**\
It will auto update when execute "./run.sh" if there are no local changes.

**How to rebuild the docker image:**\
"docker rmi image_id” (which you see using "docker images”) then "./run_docker.sh”

**If you have any additional questions feel free to ask.**
