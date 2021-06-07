vulkan_installed := $(shell command -v vulkaninfo 2> /dev/null)
glslang_installed := $(shell command -v glslangValidator 2> /dev/null)

all:
		rm -rf build
		mkdir build
ifdef vulkan_installed
ifdef glslang_installed
		cd PrivacyAmplification && $(MAKE) vulkan && cp -a PrivacyAmplification ../build
endif
endif
		cd PrivacyAmplification && $(MAKE) cuda && cp -a PrivacyAmplificationCuda ../build
		cd examples/SendKeysExample && $(MAKE) && cp -a SendKeysExample ../../build
		cd examples/MatrixSeedServerExample && $(MAKE) && cp -a MatrixSeedServerExample ../../build
		cd examples/ReceiveAmpOutExample && $(MAKE) && cp -a ReceiveAmpOutExample ../../build
		cd examples/LargeBlocksizeExample && $(MAKE) && cp -a LargeBlocksizeExample ../../build
		cp -a PrivacyAmplification/keyfile.bin ./build
		cp -a PrivacyAmplification/toeplitz_seed.bin ./build
		cp -a PrivacyAmplification/ampout.sh3 ./build
		ln PrivacyAmplification/config.yaml ./build/config.yaml

clean:
		rm -rf build
		cd PrivacyAmplification && $(MAKE) clean
		cd examples/SendKeysExample && $(MAKE) clean
		cd examples/MatrixSeedServerExample && $(MAKE) clean
		cd examples/ReceiveAmpOutExample && $(MAKE) clean
		cd examples/LargeBlocksizeExample && $(MAKE) clean
