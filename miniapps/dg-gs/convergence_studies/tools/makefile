# Builds the sampling tool used by plot_linear_errors.py against the MFEM build.
# Lives in dg-gs/convergence_studies/tools/, four levels below the MFEM root.
MFEM_DIR ?= ../../../..
MFEM_BUILD_DIR ?= ../../../..
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

TOOLS = sample_linear_GS_error

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: %.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(TOOLS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f $(TOOLS)
	rm -rf *.o
