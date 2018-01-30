'''Mash analyses'''

import os
import sys
import argparse
import numpy as np
from scipy import stats

# main code
def main():
    print("Hello\n")


###########################
# read assembly file list #
###########################

def readFile (fn):
    assemblyList = []
    try:
        ifile = open(fn,'r')
        assemblyList = [line.rstrip('\n') for line in ifile]
        ifile.close()
        return assemblyList
    except:
        sys.exit("Unable to read input file "+fn)
    return None

########################
# construct a database #
########################

def constructDatabase(refFile,klist,sketch,oPrefix):
    
    # read assemblies
    assemblyList = readFile(refFile)
    
    # create FASTA database
    fFileName = "./"+oPrefix+"/"+oPrefix+".mfa"
    fFile = open(fFileName,'w')
    for aFileName in assemblyList:
        try:
            aFile = open(aFileName,'r')
        except:
            sys.exit("Cannot open file "+aFileName)
        aContigs = aFile.readlines()
        print(">"+aFileName,file=fFile)
        for line in aContigs:
            if not line.startswith('>'):
                print(line.rstrip(),file=fFile)
        aFile.close()
    fFile.close()

    # create kmer databases
    for k in klist:
        dbname = "./"+oPrefix+"/"+oPrefix+"."+str(k)
        try:
            os.system("mash sketch -i -s "+str(sketch)+" -o "+dbname+" -k "+str(k)+" "+fFileName+" 2> /dev/null")
        except:
            sys.exit("Cannot create database "+dbname)

    # finish
    return None

####################
# query a database #
####################

def queryDatabase(qFile,klist,dbPrefix):
    
    # initialise dictionary
    raw = {}
    queryList = readFile(qFile)
    for query in queryList:
        raw[query] = {}
    
    # initialise data structures
    core = {}
    accessory = {}
    dbSketch = {}
    
    # search each query
    for k in klist:
        dbname = "./"+dbPrefix+"/"+dbPrefix+"."+str(k)+".msh"
        # get sketch size for standaridising metrics
        dbInfo = os.popen("mash info -t "+dbname).read();
        for line in dbInfo.split("\n"):
            if (line[0] != "#"):
                sketchValues = line.split("\t")
                dbSketch[str(k)] = sketchValues[0]
                break
        
        # run query
        rawOutput = os.popen("mash dist -l "+dbname+" "+qFile+" 2> "+dbPrefix+".err.log").read()
        for line in rawOutput.split("\n"):
            mashVals = line.strip().split()
            if (len(mashVals) > 2):
                mashMatch = mashVals[len(mashVals)-1].split('/')
                if (k == klist[0]):
                    raw[mashVals[1]][mashVals[0]] = {}
                raw[mashVals[1]][mashVals[0]][str(k)] = mashMatch[0]

    # run pairwise analyses
    for query in raw:
        core[query] = {}
        accessory[query] = {}
        for ref in raw[query]:
            pairwise = []
            for k in klist:
                pairwise.append(int(raw[query][ref][str(k)]))
            gradient,intercept,r_value,p_value,std_err = stats.linregress(klist,pairwise)
            core[query][ref] = float(intercept)/float(dbSketch[str(k)])
            accessory[query][ref] = -1*float(gradient)/float(dbSketch[str(k)])

    return core,accessory

##############################
# write query output to file #
##############################

def printQueryOutput(coreDict,accessoryDict,resultsPrefix):

    # open output file
    outFileName = resultsPrefix+".search.out"
    try:
        oFile = open(outFileName,'w')
    except:
        sys.exit("Cannot write to output file "+resultsPrefix+".search.out")

    # print header
    print("Query\tReference\tCore\tAccessory",file=oFile)

    # add results
    for query in coreDict:
        for ref in coreDict[query]:
            print(query+"\t"+ref+"\t"+str(coreDict[query][ref])+"\t"+str(accessoryDist[query][ref]),file=oFile)

    oFile.close()

    return None

#############################
# create database directory #
#############################

def createDatabaseDir(outPrefix):
    outputDir = os.getcwd()+"/"+outPrefix
    # check for writing
    if not os.path.isdir(outputDir):
        try:
            os.makedirs(outputDir)
        except:
            sys.exit("Cannot create output directory")

    return None

#################
# run main code #
#################

if __name__ == '__main__':
    
    # command line parsing
    parser = argparse.ArgumentParser(description='Strain structure analysis software usage:')
    parser.add_argument('-d',type = str, help='Directory containing reference database')
    parser.add_argument('-q', help='File listing query input assemblies')
    parser.add_argument('-r', help='File listing reference input assemblies')
    parser.add_argument('-m', help='Minimum kmer length (default = 19)')
    parser.add_argument('-M', help='Maximum kmer length (default = 31)')
    parser.add_argument('-s', help='Kmer sketch size (default = 10000)')
    parser.add_argument('-o', help='Prefix for output files')
    args = parser.parse_args()

    # check mash is installed
    try:
        os.system("mash > /dev/null 2> /dev/null")
    except:
        sys.exit("mash not installed on your path")

    # identify kmer properties
    minkmer = 19
    maxkmer = 31
    if args.m is not None and int(args.m) > minkmer:
        minkmer = int(args.m)
    if args.M is not None and int(args.M) < maxkmer:
        maxkmer = int(args.M)
    if minkmer >= maxkmer and minkmer >= 19 and maxkmer <= 31:
        sys.exit("Minimum kmer size "+minkmer+" must be smaller than maximum kmer size "+maxkmer+"; range must be between 19 and 31")
    kmers = np.arange(minkmer,maxkmer+1,2)
    sketchSize = 10000
    if args.s is not None:
        sketchSize = arg.s

    # check on output prefix
    if args.o is None:
        sys.exit("Please provide an output file prefix")
    
    # determine mode for running
    coreDist = {}
    accessoryDist = {}
    if args.r is not None:
        if args.d is None and args.q is None:
            print("Constructing database from reference assemblies in "+args.r+"\n")
            createDatabaseDir(args.o)
            constructDatabase(args.r,kmers,sketchSize,args.o)
            coreDist,accessoryDist = queryDatabase(args.r,kmers,args.o)
            printQueryOutput(coreDist,accessoryDist,args.o)
        else:
            sys.exit("Do not provide database name with '-d' when creating a database")
    elif args.d is not None:
        if args.q is not None:
            print("Querying database "+args.d+" with assemblies in "+args.q+"\n")
            coreDist,accessoryDist = queryDatabase(args.q,kmers,args.d)
            printQueryOutput(coreDist,accessoryDist,args.o)
        else:
            sys.exit("Need both a database and query file to run a search")
    
    # main processing
    main()