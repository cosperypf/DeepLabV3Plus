
all: clean

clean:
	find ./ | grep .DS_Store | xargs rm -f
	find ./ | grep __pycache__ | xargs rm -rf
	rm -rf ./results ./models
