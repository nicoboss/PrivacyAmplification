CFLAGS = -O2 -g -Wall
CXXFLAGS = -O2 -g -Wall -std=c++17
LIBS = -pthread -lzmq -L/opt/rocm/lib -lrocfft

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

all:  PrivacyAmplification_hip.o yaml/Yaml.o  sha3/sha3.o PrivacyAmplification_hip

PrivacyAmplification_hip.o: PrivacyAmplification_hip.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@
	
yaml/Yaml.o:yaml/Yaml.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

sha3/sha3.o:sha3/sha3.c
	$(HIPCC) $(CFLAGS) -c $< -o $@

PrivacyAmplification_hip:
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	$(HIPCC) $(CXXFLAGS) PrivacyAmplification_hip.o yaml/Yaml.o sha3/sha3.o -lm -o $@ $(LIBS)
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

.PHONY: clean

clean:
	rm -f PrivacyAmplification_hip PrivacyAmplification_hip.o yaml/Yaml.o sha3/sha3.o
