#module MemNN
using Knet

#Config
trainmod=true
memoryLength=14
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
dict = Dict{ASCIIString,Float64}()
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
memAry = zeros(Float64,length(dict),memoryLength)
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
    #xmemVec = Float64[]
    #append!(xmemVec,x)
    #append!(xmemVec,vec(memAry))
    
    if(trainmod)
        train_o1(o1_costmodel, x, memAry, y_memloc, softloss,size(memAry,2))# related memory location probability maximization training
    end
    return forw(o1_costmodel, x, memAry;mem_length=size(memAry,2)) #returns probabilities of memorylocations
end #O_module

function R_module(mem1, x, y_answordloc, dict, trainmod,r_costmodel)

    xansVec = Float64[]
    append!(xansVec,x)
    append!(xansVec,mem1)
    append!(xansVec,zeros(length(dict)))
    
    words = collect(keys(dict))
    dictVec = collect(values(dict))
    dictMatrix = zeros(Float64,length(words)*3,length(dict))
    for(ind=1:length(dict))
        dictMatrix[ind+length(words)*2,ind] = 1
    end
    
    if(trainmod)
        train_r(r_costmodel, xansVec, dictMatrix, y_answordloc, softloss, length(words))# related memory location probability maximization training
    end
    ans_vec = forw(r_costmodel, xansVec, dictMatrix ;dict_length=length(words))
    answer = words[prob_vec2indice(ans_vec)] # finds indice of maximum value from the vector of probabilities. 
    return answer
    
end #R_module





#############################################################################################
#Auxilary functions
    @knet function o1_cost(x,memAry; winit=Gaussian(0,.1),mem_length=14,pdrop=0.5,lr=0.001,a=50)
        u = par(init=winit, dims=(a,0))#19
        m = transp(memAry)*transp(u)
        n = u*x
        r = m*n
    #fndrop = drop(t; pdrop=pdrop)
    #t = repeat(x; frepeat=:wbf, nrepeat=10, out=30,f=:relu,winit=winit)
    t=wbf(r; out=mem_length, f=:soft, winit=winit)
    return t
end
@knet function r_cost(x, dictMatrix ; winit=Gaussian(0,.1),dict_length=19,a=100)
    u = par(init=winit, dims=(a,0))
        m = transp(dictMatrix)*transp(u)
        n = u*x
    j = m*n
    #t = wbf(j; out=30, f=:relu, winit=winit)
    #k = wbf(t; out=30, f=:relu, winit=winit)
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
function train_o1(o1_costmodel, x , memAry, y_memloc, loss, length)
    forw(o1_costmodel,x, memAry;mem_length=length,dropout=false)
    #forw(o1_costmodel,xmemVec)
    back(o1_costmodel,y_memloc, loss)
    update!(o1_costmodel)
end
function train_r(r_costmodel, xansVec,dictMatrix, y_ansloc, loss, length)
    forw(r_costmodel,xansVec,dictMatrix;dict_length=length)
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

function prob_vec2indice(words;size_limit=0)
    max=0
    pred_ans_loc = 1
    if(size_limit==0)
        limit=length(words)
    else
        limit=size_limit
        end
    for w = 1:limit
        if(words[w]>max)
            max = words[w]
            pred_ans_loc = w
        end
        end
        return pred_ans_loc
end
        
#############################################################################################
    o1_costmodel = compile(:o1_cost) #Knet Model Compilation for o1_costmodel
    r_costmodel = compile(:r_cost)   #Knet Model Compilation for r_costmodel
    olr=0.0001 # sets initial lr for training o1_costmodel(for decaying lr)
    setp(o1_costmodel, lr=olr)
    setp(r_costmodel, lr=0.00001) # sets initial lr for training r_costmodel
    old=0 #for tracking 1 step older test performance
 ## Main Flow
 ############################################################################################
   #EPOCH Loop
    for(k=1:100)
       # println("epoch",k)
        trainmod=true # TRAIN/TEST switch
        memCount=1    #Counter for memory location
        
        #initialisations for statistics
        trquestioncount=0
        trsum=0
        tr_correct_response_count=0
        test_correct_response_count=0
        #################################
   #Train Data Loop     
    for i=1:size(data,1)
        sen=data[i,1]
        ans=data[i,2]
        clu=data[i,3]
        newMem = I_module(sen[2],dict)
        
        if(sen[1]=="1") #new story
            memCount=1
            memAry = zeros(Float64,length(dict),memoryLength)

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
            memCount+=1
            y_memloc=zeros(size(memAry,2))
            y_memloc[clu]=1
            trquestioncount+=1
            memLoc=O_module(memAry,newMem,y_memloc,trainmod,o1_costmodel)
           #println("memLoc",memLoc)
            #println("clu",clu)
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
            #println("vector of R module response: ",get(r_costmodel,:r))
               if(f_answer==ans)
                   tr_correct_response_count+=1
                   end
        end#question
            

        end#data iterator(for)
           # if((trsum/trquestioncount)>0.5)
           #     setp(o1_costmodel, pdrop=0.77)
           # end
                
#print(k,"\t",trsum/trquestioncount,"\t")          
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
            memAry = zeros(Float64,length(dict),memoryLength)

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
            memCount+=1
            questionquantity+=1
            y_memloc=zeros(size(memAry,2))
            y_memloc[clu]=1
            
            memLoc=O_module(memAry,newMem,y_memloc,trainmod,o1_costmodel)
            #println("test:memLoc",memLoc)
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
                if(f_answer==ans)
                   test_correct_response_count+=1
                   end
        end#question

        end#test_data iterator(for)
           
            println(k,"\t",trsum/trquestioncount,"\t",sum/questionquantity,"\t R_Module training: ",tr_correct_response_count/trquestioncount,"\t test:",test_correct_response_count/questionquantity)
            #println(k,"\t",trsum/trquestioncount,"\t",sum/questionquantity,"\t",tr_correct_response_count/trquestioncount,"\t",test_correct_response_count/questionquantity)
            
            #For Decaying lr by o1_costmodel
            ###########################################
            if(old>(sum/questionquantity))
                
                olr =0.8 * olr
                setp(o1_costmodel,lr=olr)
               
               setp(o1_costmodel, pdrop=0.50)
            
            end
                old=sum/questionquantity
            ###########################################
end#epochs

            ##


    ##
#end #module
