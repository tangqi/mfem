# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/dg-gs/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# The single-file DG fixed-boundary Grad-Shafranov solver
MINIAPPS = DG_fixed_boundary_linear_GS

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean
.PRECIOUS: %.o

# Remove built-in rules
%: %.cpp
%.o: %.cpp

%: %.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $^ -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

all: $(MINIAPPS)

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f $(MINIAPPS)
	rm -rf *.o
	rm -f mesh.* sol.*
