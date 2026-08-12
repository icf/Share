import subprocess
import sys
import os

if len(sys.argv) < 2:
    sys.exit("Missing arguments!!! Example: python dataAnalyze.py blockSize")
blockSize  = int( sys.argv[1] )

L = 28 * 2

if os.path.isfile('./HNum_initial.dat'):
    print( "\033[1m" "Calculate HAverage." "\033[0m" )
    subprocess.call( os.environ['AFQMCLAB_DIR']+"/bin/NumDenErrorAnalysis HNum_initial.dat den_initial.dat HAverage_initial.dat", shell=True)

if os.path.isfile('./greenNum_initial.dat'):
    print( "\033[1m" "Calculate greenMatrixAverage." "\033[0m" )
    subprocess.call( os.environ['AFQMCLAB_DIR']+
                     "/bin/NumArrayDenErrorAnalysis greenNum_initial.dat den_initial.dat greenMatrix_initial_Average.dat {0:d} {1:d}".format(L*L, blockSize),shell=True)

if os.path.isfile('./greenNum.dat'):
    print( "\033[1m" "Calculate greenMatrixAverage." "\033[0m" )
    subprocess.call( os.environ['AFQMCLAB_DIR']+
                     "/bin/NumArrayDenErrorAnalysis greenNum.dat den.dat greenMatrix_Average.dat {0:d} {1:d}".format(L*L, blockSize),shell=True)

if os.path.isfile('./HNum0.dat'):
    print( "\033[1m" "Calculate HAverage." "\033[0m" )
    subprocess.call( os.environ['AFQMCLAB_DIR']+"/bin/NumDenErrorAnalysis HNum0.dat den0.dat HAverage0.dat", shell=True)

if os.path.isfile('./signRatioNum0.dat'):
    print( "\033[1m" "Calculate signRatioAverage0." "\033[0m" )
    subprocess.call( os.environ['AFQMCLAB_DIR']+"/bin/NumDenErrorAnalysis signRatioNum0.dat den0.dat signRatioAverage0.dat", shell=True)


if os.path.isfile('./HNum.dat'):
    print( "\033[1m" "Calculate HAverage." "\033[0m" )
    subprocess.call( os.environ['AFQMCLAB_DIR']+"/bin/NumDenErrorAnalysis HNum.dat den.dat HAverage.dat", shell=True)

if os.path.isfile('./HNum_SDSD.dat'):
    print( "\033[1m" "Calculate HAverage_SDSD." "\033[0m" )
    subprocess.call( os.environ['AFQMCLAB_DIR']+"/bin/NumDenErrorAnalysis HNum_SDSD.dat den_SDSD.dat HAverage_SDSD.dat", shell=True)


