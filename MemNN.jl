#module MemNN
using Knet

#Config
trainmod=true
#



### Data parsing
data = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa1_single-supporting-fact_test.txt",'\t')
for i=1:size(data,1)
    data[i,1] = split(data[i,1],' ',limit=2,keep=true)
    #format: SubString{ASCIIString}["3","Where is John? "]                 "hallway"    1  
end
###

#Dictionary initialization (adding each word in dataset to dictionary.)
dict = Dict{ASCIIString,Int}()
dictdata = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa1_single-supporting-fact_test.txt",' ')
for s in dictdata
    if(typeof(s)!=Int64&&s!="")
        if(contains(s,"\t"))
           s=strip(s,['?','!',',','.','\t','1','2','3','4','5','6','7','8','9','0']) # ? char is undesired near the words
           else
           s=strip(s,['?','!',',','.'])
           end
        get!(dict,s,0)
    end
end
#

# Memory Initialization
memAry = Array(Float64,length(dict),size(data,1))
#

#########################################################################################################
# Modules
    function I_module(str,dict) #input:sentence,dictionary
     dictn=copy(dict)   
    spltstr = split(str) #sentence tokenize
        for s in spltstr
            s=strip(s,['?','!',',','.','\t'])
        dictn[s]=(dictn[s]+1) # going over each word and updating dict
    end
    return collect(values(dictn)) # array of BoW
end #I_module

function G_module(memAry,newMem,memCount)
    
    memAry[:,memCount] = newMem   #updates memory by adding BoW vectors to the list of memories
end #G_module

function O_module(memAry,x,y_memloc,trainmod)
    # by this configuration, the module needs supporting facts, which are locations of the related memories as onehot vec.
 #sizehint for memVec to be enough long(later-for performance)
    xmemVec = Float64[]
    append!(xmemVec,x)
    append!(xmemVec,vec(memAry))
    o1_costmodel = compile(:o1_cost)
    if(trainmod)
        train_o1(o1_costmodel, xmemVec, y_memloc, softloss,length(memAry))# related memory location probability maximization training
    end
    return forw(o1_costmodel, xmemVec;mem_length=length(memAry)) #returns probabilities of memorylocations
end #O_module

function R_module(memLoc, x, y_answordloc, dict, trainmod)

    xansVec = Float64[]
    append!(xansVec,x)
    append!(xansVec,memLoc)
    r_costmodel = compile(:r_cost)
    words = collect(keys(dict))
    if(trainmod)
        train_r(r_costmodel, xansVec, y_answordloc, softloss, length(words))# related memory location probability maximization training
    end
    ans_vec = forw(r_costmodel, xansVec;dict_length=length(words))
    answer = words[prob_vec2indice(ans_vec)] # finds indice of maximum value from the vector of probabilities. 
    return answer
    
end #R_module





#############################################################################################
#Auxilary functions
@knet function o1_cost(x; winit=Gaussian(0,.1),mem_length=57000)
    h    = wbf(x; out=1, f=:copy, winit=winit)
    j    = wbf(h; out=1, f=:copy, winit=winit)
    return wbf(j; out=mem_length, f=:soft, winit=winit)
end
@knet function r_cost(x; winit=Gaussian(0,.1),dict_length=19)
    h    = wbf(x; out=1, f=:copy, winit=winit)
    j    = wbf(h; out=1, f=:copy, winit=winit)
    return wbf(j; out=dict_length, f=:soft, winit=winit)
end

function train(f, data, loss)
    for (x,y) in data
        forw(f,x)
        back(f,y,loss)
        update!(f)
    end
end
function train_o1(o1_costmodel, xmemVec, y_memloc, loss, length)
    forw(o1_costmodel,xmemVec;mem_length=length)
    #forw(o1_costmodel,xmemVec)
    back(o1_costmodel,y_memloc, loss)
    update!(o1_costmodel)
end
function train_r(r_costmodel, xansVec, y_ansloc, loss, length)
    forw(r_costmodel,xansVec;dict_length=length)
     #forw(r_costmodel,xansVec)
    back(r_costmodel,y_ansloc, loss)
    update!(r_costmodel)
end

function test(f,data,loss)
    sumloss = numloss = 0
    for(x,ygold) in data
        ypred = forw(f,x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
        sumloss /numloss
    end

function prob_vec2indice(words)
    max=0
    pred_ans_loc = 0
    for w = 1:length(words)
        if(words[w]>max)
            max = words[w]
            pred_ans_loc = w
        end
        end
        return pred_ans_loc
end
        
#############################################################################################


    ## Main Flow
    memCount=1
    for i=1:size(data,1)
        sen=data[i,1]
        ans=data[i,2]
        clu=data[i,3]
        newMem = I_module(sen[2],dict)
        
        if(sen[1]=="1") #new story
            memCount=1
            memAry = Array(Float64,length(dict),size(data,1))

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
            y_memloc=zeros(length(memAry))
            y_memloc[clu]=1
            
            memLoc=O_module(memAry,newMem,y_memloc,trainmod)
            println(length(memLoc))
            dictn = copy(dict)
            dictn[ans] = 1
           println(memLoc)
            y_answordloc = collect(values(dictn))
            
            f_answer=R_module(vec(memLoc), newMem, y_answordloc, dict, trainmod)
            println("Answer : ",f_answer) 
            println("RightAnswer : ",ans,".")


        end#question

            end#data iterator(for)



    ##
#end #module
