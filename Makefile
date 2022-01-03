all:
	cd ic
	make
	cd ..
	cd experiments
	quarto render

