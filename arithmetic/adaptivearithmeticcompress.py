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
from . import arithmeticcoding
python3 = sys.version_info.major >= 3
import numpy as np


def compress(snp_list, mask_list, numsymbol):
	initfreqs = arithmeticcoding.FlatFrequencyTable(numsymbol)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(32)

	for len_list in range(len(snp_list)):
		snp = snp_list[len_list]
		mask = mask_list[len_list]

		snp  = np.squeeze(snp,axis=0)
		mask = np.squeeze(mask,axis=0)
		rows,cols,channel = snp.shape
	
		for c in range(channel):      
			for i in range(rows):
				for j in range(cols):
					if c==0:
						symbol = snp[i,j,c]	 
						enc.write(freqs, symbol)
						freqs.increment(symbol)
					elif mask[i,j]==1:
						symbol = snp[i,j,c]	 
						enc.write(freqs, symbol)
						freqs.increment(symbol)
	enc.write(freqs, numsymbol-1)  # EOF
	enc.finish()  # Flush remaining code bits
	return enc.bit_nums# + rows*channel


def ori_compress(snp_list, numsymbol):
	initfreqs = arithmeticcoding.FlatFrequencyTable(numsymbol)
	freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
	enc = arithmeticcoding.ArithmeticEncoder(32)
	for len_list in range(len(snp_list)):
		snp = snp_list[len_list]
		#snp  = snp.as
		
		snp  = snp.reshape(1,-1)
		snp  = np.squeeze(snp)
		#mask = np.squeeze(mask,axis=0)
		length = len(snp)
		#print(snp.shape)
		for i in range(length):      
		
				
			symbol = snp[i]	 
			enc.write(freqs, symbol)
			freqs.increment(symbol)
				
	enc.write(freqs, numsymbol-1)  # EOF
	enc.finish()  # Flush remaining code bits
	return enc.bit_nums# + rows*channel



# Main launcher
if __name__ == "__main__":
	main(sys.argv[1 : ])
