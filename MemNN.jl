#module MemNN
using Knet

#Config
trainmod=true
#



### Training Data parsing
data = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa1_single-supporting-fact_train.txt",'\t')
for i=1:size(data,1)
    data[i,1] = split(data[i,1],' ',limit=2,keep=true)
    #format: SubString{ASCIIString}["3","Where is John? "]                 "hallway"    1  
end
###

### Test Data parsing
test_data = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa1_single-supporting-fact_test.txt",'\t')
for i=1:size(test_data,1)
    test_data[i,1] = split(test_data[i,1],' ',limit=2,keep=true)
    #format: SubString{ASCIIString}["3","Where is John? "]                 "hallway"    1  
end
###

#Dictionary initialization (adding each word in dataset to dictionary.)
dict = Dict{ASCIIString,Int}()
dictdata = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa1_single-supporting-fact_train.txt",' ')
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
memAry = zeros(Float64,length(dict),14)
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

function O_module(memAry,x,y_memloc,trainmod,o1_costmodel)
    # by this configuration, the module needs supporting facts, which are locations of the related memories as onehot vec.
 #sizehint for memVec to be enough long(later-for performance)
    xmemVec = Float64[]
    append!(xmemVec,x)
    append!(xmemVec,vec(memAry))
    
    if(trainmod)
        train_o1(o1_costmodel, xmemVec, y_memloc, softloss,size(memAry,2))# related memory location probability maximization training
    end
    return forw(o1_costmodel, xmemVec;mem_length=size(memAry,2)) #returns probabilities of memorylocations
end #O_module

function R_module(mem1, x, y_answordloc, dict, trainmod,r_costmodel)

    xansVec = Float64[]
    append!(xansVec,x)
    append!(xansVec,mem1)
    
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
@knet function o1_cost(x; winit=Gaussian(0,.1),mem_length=14)
    h    = wbf(x; out=50, f=:relu, winit=winit)
    j    = wbf(h; out=50, f=:relu, winit=winit)
    l    = wbf(j; out=50, f=:relu, winit=winit)
    t    = wbf(l; out=50, f=:relu, winit=winit)
    #t = repeat(x; frepeat=:wbf, nrepeat=10, out=30,f=:relu,winit=winit)
    return wbf(t; out=mem_length, f=:soft, winit=winit)
end
@knet function r_cost(x; winit=Gaussian(0,.1),dict_length=19)
    h    = wbf(x; out=30, f=:relu, winit=winit)
    j    = wbf(h; out=30, f=:relu, winit=winit)
    r = wbf(j; out=dict_length, f=:soft, winit=winit)
    return r
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
    o1_costmodel = compile(:o1_cost)
    r_costmodel = compile(:r_cost)
    setp(o1_costmodel, lr=0.001)
    ## Main Flow
    for(k=1:60)
       # println("epoch",k)
        trainmod=true
        memCount=1
        trquestioncount=0
        trsum=0
    for i=1:size(data,1)
        sen=data[i,1]
        ans=data[i,2]
        clu=data[i,3]
        newMem = I_module(sen[2],dict)
        
        if(sen[1]=="1") #new story
            memCount=1
            memAry = zeros(Float64,length(dict),14)

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
            y_memloc=zeros(size(memAry,2))
            y_memloc[clu]=1
            trquestioncount+=1
            memLoc=O_module(memAry,newMem,y_memloc,trainmod,o1_costmodel)
           
            dictn = copy(dict)
            dictn[ans] = 1
           if(clu==prob_vec2indice(memLoc))
               trsum+=1
               end
            y_answordloc = collect(values(dictn))
            #println(memAry[:,prob_vec2indice(memLoc)])
            #f_answer=R_module(memAry[:,prob_vec2indice(memLoc)], newMem, y_answordloc, dict, trainmod,r_costmodel)
            f_answer=R_module(memAry[:,clu], newMem, y_answordloc, dict, trainmod,r_costmodel)
            #println(i)
            #println("Answer : ",f_answer) 
            #println("RightAnswer : ",ans,".")
            

        end#question
            

        end#data iterator(for)
print(trsum/trquestioncount,"\t")          
           ##Test
            trainmod=false
        memCount=1
            sum=0
            questionquantity=0
    for i=1:size(test_data,1)
        sen=test_data[i,1]
        ans=test_data[i,2]
        clu=test_data[i,3]
        newMem = I_module(sen[2],dict)
        
        if(sen[1]=="1") #new story
            memCount=1
            memAry = zeros(Float64,length(dict),14)

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
            questionquantity+=1
            y_memloc=zeros(size(memAry,2))
            y_memloc[clu]=1
            
            memLoc=O_module(memAry,newMem,y_memloc,trainmod,o1_costmodel)
            #println("memLoc",memLoc)
            #println("clu",clu)
            dictn = copy(dict)
            dictn[ans] = 1
            if(clu==prob_vec2indice(memLoc))
                sum+=1
            #else
            #    println("thiiis")
                end
            y_answordloc = collect(values(dictn))
            
            f_answer=R_module(memAry[:,clu], newMem, y_answordloc, dict, trainmod,r_costmodel)
            #f_answer=R_module(memAry[:,prob_vec2indice(memLoc)], newMem, y_answordloc, dict, trainmod,r_costmodel)
            #println("Answer : ",f_answer) 
            #println("RightAnswer : ",ans,".")
            #println("vector of R module response: ",get(r_costmodel,:r))

        end#question

            end#test_data iterator(for)
println(sum/questionquantity)
end#epochs

            ##


    ##
#end #module
