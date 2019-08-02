# 
# Compression application using adaptive arithmetic coding
# 
# Usage: python adaptive-arithmetic-compress.py InputFile OutputFile
# Then use the corresponding adaptive-arithmetic-decompress.py application to recreate the original input file.
# Note that the application starts with a flat frequency table of 257 symbols (all set to a frequency of 1),
# and updates it after each byte encoded. The corresponding decompressor program also starts with a flat
# frequency table and updates it after each byte decoded. It is by design that the compressor and
# decompressor have synchronized states, so that the data can be decompressed properly.
# 
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 

import contextlib, sys
import arithmeticcoding
python3 = sys.version_info.major >= 3
import numpy as np



def compress(snp, numsymbol):
	initfreqs = arithmeticcoding.FlatFrequencyTable(numsymbol)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(32)

	snp  = np.squeeze(snp)
	rows,cols,channel = snp.shape
	
	for c in range(channel):      
		for i in range(rows):
			for j in range(cols):
		# Read and encode one byte
				symbol = snp[i,j,c]	 
				enc.write(freqs, symbol)
				freqs.increment(symbol)
	enc.write(freqs, numsymbol-1)  # EOF
	enc.finish()  # Flush remaining code bits
	return enc.bit_nums


# Main launcher
if __name__ == "__main__":
	testarray = np.random.randint(low=5,size=[1,50,50,2])
	print(compress(testarray,5))
	print(testarray-1)
