CXXFLAGS += -std=c++11 -O3

all:  PrivacyAmplification_hip

common.o: common.cpp common.h Makefile

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

PrivacyAmplification_hip.o : PrivacyAmplification_hip.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

PrivacyAmplification_hip: PrivacyAmplification_hip.o Makefile
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	$(HIPCC) $(CXXFLAGS) $< -lm -o $@
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif


.PHONY: clean

clean:
	rm -f PrivacyAmplification_hip *.o

