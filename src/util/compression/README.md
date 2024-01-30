```
Testing compression ratio: 1000 docs, size:
min                    24.43Kb
max                    49.37Kb
avg                     36.6Kb
median                 36.23Kb
pc75                   43.16Kb
pc90                   46.96Kb
all                    35.74Mb
BrotliCompressor(q=1)     CR 2.582 | compression     251.46Mb/s | decompression   317.54Mb/s | compressed avg: 14.17Kb
BrotliCompressor(q=2)     CR 2.786 | compression     128.67Mb/s | decompression   357.43Mb/s | compressed avg: 13.13Kb
BrotliCompressor(q=3)     CR 2.807 | compression     110.76Mb/s | decompression   378.19Mb/s | compressed avg: 13.04Kb
BrotliCompressor(q=4)     CR 2.875 | compression      66.64Mb/s | decompression   353.69Mb/s | compressed avg: 12.72Kb
BrotliCompressor(q=6)     CR 3.041 | compression      43.45Mb/s | decompression   356.92Mb/s | compressed avg: 12.03Kb
BrotliCompressor(q=8)     CR 3.051 | compression       33.9Mb/s | decompression   355.68Mb/s | compressed avg: 11.99Kb
BrotliCompressor(q=11)    CR 3.433 | compression     934.31Kb/s | decompression   275.57Mb/s | compressed avg: 10.65Kb
ZLibCompressor            CR 2.922 | compression      48.67Mb/s | decompression   288.72Mb/s | compressed avg: 12.52Kb
Bzip2Compressor           CR 3.009 | compression      15.89Mb/s | decompression    44.18Mb/s | compressed avg: 12.14Kb
LzmaCompressor            CR 3.111 | compression       4.92Mb/s | decompression    54.98Mb/s | compressed avg: 11.75Kb

Testing compression ratio: 6 docs, size:
min                    20.51Kb
max                    541.6Kb
avg                   246.36Kb
median                 165.3Kb
pc75                  360.32Kb
pc90                  483.16Kb
all                     1.44Mb
BrotliCompressor(q=1)     CR 4.664 | compression     350.84Mb/s | decompression   473.42Mb/s | compressed avg: 49.88Kb
BrotliCompressor(q=2)     CR 5.351 | compression     179.02Mb/s | decompression   569.98Mb/s | compressed avg: 43.96Kb
BrotliCompressor(q=3)     CR 5.622 | compression     163.57Mb/s | decompression   643.73Mb/s | compressed avg: 42.06Kb
BrotliCompressor(q=4)     CR 5.898 | compression     130.84Mb/s | decompression   733.19Mb/s | compressed avg: 40.22Kb
BrotliCompressor(q=6)     CR 6.916 | compression      63.75Mb/s | decompression   730.68Mb/s | compressed avg: 34.34Kb
BrotliCompressor(q=8)     CR 7.138 | compression      38.42Mb/s | decompression    773.7Mb/s | compressed avg: 33.21Kb
BrotliCompressor(q=11)    CR 8.44  | compression     779.77Kb/s | decompression   680.97Mb/s | compressed avg: 27.76Kb
BrotliCompressor(q=11)    CR 8.44  | compression     780.29Kb/s | decompression   683.45Mb/s | compressed avg: 27.76Kb
ZLibCompressor            CR 5.945 | compression      56.26Mb/s | decompression   493.85Mb/s | compressed avg: 40.37Kb
Bzip2Compressor           CR 8.05  | compression      16.32Mb/s | decompression    61.91Mb/s | compressed avg: 29.38Kb
LzmaCompressor            CR 8.209 | compression       5.18Mb/s | decompression   136.37Mb/s | compressed avg: 28.47Kb
```
