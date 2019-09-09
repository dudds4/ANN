RM = rm -f

CC = g++
CFLAGS = -O2 -std=c++17 -g

LIBHDRS = inc/layer.h inc/neuralnet.h inc/activationTypes.h inc/math_help.h
LIBSRCS = src/layer.cpp src/neuralnet.cpp src/math_help.cpp

INCLUDES = inc

MSRCS = main.cpp
TSRCS = tests.cpp

all: bin bin/toyml bin/tests

bin:
	mkdir bin

bin/toyml: ${MSRCS} ${LIBHDRS} ${LIBSRCS}
# 	@printf "Compiling toyml\n"
	$(CC) -I${INCLUDES} ${MSRCS} ${LIBSRCS} $(CFLAGS) -o bin/toyml

bin/tests: ${TSRCS} ${LIBHDRS} ${LIBSRCS}
# 	@printf "Compiling tests\n"
	$(CC) -I${INCLUDES} ${TSRCS} ${LIBSRCS} $(CFLAGS) -o bin/tests

clean:
	$(RM) -r bin
