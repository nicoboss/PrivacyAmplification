all:
	rm -rf build
	mkdir build
	cd PrivacyAmplification && $(MAKE) && cp -a PrivacyAmplification ../build
	cd examples/SendKeysExample && $(MAKE) && cp -a SendKeysExample ../../build
	cd examples/MatrixSeedServerExample && $(MAKE) && cp -a MatrixSeedServerExample ../../build
	cd examples/ReceiveAmpOutExample && $(MAKE) && cp -a ReceiveAmpOutExample ../../build
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
