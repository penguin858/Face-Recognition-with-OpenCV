LocalBinary:LocalBinary.cpp
	g++ -o eigen_carg eigen_carg.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv` 
	g++ -o fisher_carg fisher_carg.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv` 
	g++ -o LocalBinary LocalBinary.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv` 

clean:
	rm LocalBinary
