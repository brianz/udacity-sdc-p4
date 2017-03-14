.PHONY:	all 

all:
	docker run --rm -it \
		-p 8888:8888 -v `pwd`:/src \
		udacity/carnd-term1-starter-kit
