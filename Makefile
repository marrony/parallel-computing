LDFLAGS=-framework OpenCL
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

DIR=bin
BINS=$(addprefix $(DIR)/, $(basename $(SRCS)))

bin/%: %.c
	@mkdir -p $(DIR)
	gcc -o $@ $< $(LDFLAGS) $(ARGS)

all: $(BINS)

#.PHONY: bin

#%.o: %.c $(basename $(OBJS))
#   gcc -o $(basename $<) $< $(LDFLAGS)
