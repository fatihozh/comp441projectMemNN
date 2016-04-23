

using Knet

#Config
trainmod=true
global memoryLength=19
#



### Training Data parsing
data = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa14_time-reasoning_train.txt",'\t')
for i=1:size(data,1)
    data[i,1] = split(data[i,1],' ',limit=2,keep=true)
    #format: SubString{ASCIIString}["3","Where is John? "]                 "hallway"    1  
end
###

### Test Data parsing
test_data = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa14_time-reasoning_test.txt",'\t')
for i=1:size(test_data,1)
    test_data[i,1] = split(test_data[i,1],' ',limit=2,keep=true)
    #format: SubString{ASCIIString}["3","Where is John? "]                 "hallway"    1  
end
###

#Dictionary initialization (adding each word in dataset to dictionary.)
dict = Dict{ASCIIString,Float64}()
#dictdata = readdlm("/Users/fatihozhamaratli/Downloads/tasksv11/en/qa17_positional-reasoning_train.txt",' ')

#for s in dictdata
#    if(typeof(s)!=Int64&&s!="")
#        if(contains(s,"\t"))
#           s=strip(s,['?','!',',','.','\t','1','2','3','4','5','6','7','8','9','0']) # ? char is undesired near the words
#           else
#           s=strip(s,['?','!',',','.'])
#           end
#        get!(dict,s,0)
#    end
#end
for i=1:size(data,1)
    s=data[i,1][2]
if(typeof(s)!=Int64&&s!="")
        if(contains(s,"\t"))
           s=strip(s,['?','!',',','.','\t','1','2','3','4','5','6','7','8','9','0']) # ? char is undesired near the words
           else
           s=strip(s,['?','!',',','.'])
        end
            for t in split(s,' ')
        get!(dict,lowercase(t),0)
        end
        end
    s=data[i,2]
if(s!="")
        if(contains(s,"\t"))
           s=strip(s,['?','!',',','.','\t','1','2','3','4','5','6','7','8','9','0']) # ? char is undesired near the words
           else
           s=strip(s,['?','!',',','.'])
        end
            for t in split(s,' ')
        get!(dict,lowercase(t),0)
        end
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
            s=strip(lowercase(s),['?','!',',','.','\t'])

            dictn[s]=(dictn[s]+1) # going over each word and updating dict
    end
    return collect(values(dictn)) # array of BoW
end #I_module

function G_module(memAry,newMem,memCount)
    
    memAry[:,memCount] = newMem   #updates memory by adding BoW vectors to the list of memories
end #G_module

function O_module(memAry,x,y_memloc,y2_memloc,clu,trainmod,o1_costmodel,memCount)
    # by this configuration, the module needs supporting facts, which are locations of the related memories as onehot vec.
 #sizehint for memVec to be enough long(later-for performance)
    #xmemVec = Float64[]
    #append!(xmemVec,x)
    #append!(xmemVec,vec(memAry))
    xp =placement(x,memCount,1)
   global memAryp= placement(memAry,1,3)
    x2p =placement(x,0,1)+placement(memAry[:,clu],memCount,2)
    if(trainmod)
        train_o1(o1_costmodel, xp, memAryp, y_memloc, softloss,size(memAry,2))# related memory location probability maximization training
        train_o2(o2_costmodel, x2p, memAryp, y2_memloc, softloss,size(memAry,2))
    end
    return forw(o1_costmodel, xp, memAryp;mem_length=size(memAry,2)),forw(o2_costmodel, x2p, memAryp;mem_length=size(memAry,2)) #returns probabilities of memorylocations
end #O_module

function R_module(mem1,mem2, x, y_answordloc, dict, trainmod,r_costmodel)

    #xansVec = Float64[]
    #append!(xansVec,x)
    #append!(xansVec,mem1)
    #append!(xansVec,zeros(length(dict)))

    xp=placement(x,0,1)+placement(mem1,0,2)+placement(mem2,0,3)
    
    
    words = collect(keys(dict))
    dictVec = collect(values(dict))
    dictMatrix = zeros(Float64,length(words)*4+memoryLength,length(dict))
    for(ind=1:length(dict))
        dictMatrix[ind+length(words)*3,ind] = 1
    end
    
    if(trainmod)
        train_r(r_costmodel, xp, dictMatrix, y_answordloc, softloss, length(words))# related memory location probability maximization training
    end
    ans_vec = forw(r_costmodel, xp, dictMatrix ;dict_length=length(words))
    answer = words[prob_vec2indice(ans_vec)] # finds indice of maximum value from the vector of probabilities. 
    return answer
    
end #R_module





#############################################################################################
#Auxilary functions
    @knet function o1_cost(x,memAry; winit=Gaussian(0,.1),mem_length=0,pdrop=0.5,lr=0.001,a=100,l2reg=false)
        u = par(init=winit, dims=(a,0))#19
        m = transp(memAry)*transp(u)
        n = u*x
        r = m*n
    #fndrop = drop(t; pdrop=pdrop)
        #t = repeat(x; frepeat=:wbf, nrepeat=10, out=30,f=:relu,winit=winit)
        #k = wbf(r; out=50, f=:relu, winit=winit)
       
    t=wbf(r; out=mem_length, f=:soft, winit=winit)
    return t
    end
         @knet function o2_cost(x,memAry; winit=Gaussian(0,.1),mem_length=0,pdrop=0.5,lr=0.0001,a=100,l2reg=false)
        u = par(init=winit, dims=(a,0))#19
        m = transp(memAry)*transp(u)
        n = u*x
        r = m*n
    #k = wbf(r; out=30, f=:relu, winit=winit)
    t=wbf(r; out=mem_length, f=:soft, winit=winit)
    return t
end
@knet function r_cost(x, dictMatrix ; winit=Gaussian(0,.1),dict_length=32,a=100)
    u = par(init=winit, dims=(a,0))
        m = transp(dictMatrix)*transp(u)
        n = u*x
    j = m*n
    #t = wbf(j; out=30, f=:relu, winit=winit)
   # k = wbf(j; out=30, f=:relu, winit=winit)
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
        function train_o2(o2_costmodel, x , memAry, y2_memloc, loss, length)
    forw(o2_costmodel,x, memAry;mem_length=length,dropout=false)
    #forw(o1_costmodel,xmemVec)
    back(o2_costmodel,y2_memloc, loss)
    update!(o2_costmodel)
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
        function placement(vec_Mat,memCount,slot)

        vm=zeros(length(dict)*4+memoryLength,size(vec_Mat,2))
        
            
                if(size(vec_Mat,2)==1)
                   vm=zeros(length(vec_Mat)*4 + memoryLength)
        #Slot arrangement
                    vm[(slot-1)*length(dict)+1:slot*length(dict)]=vec_Mat
                    if(memCount!=0)
                        vm[4*length(dict)+memCount]=1
                        end
                   else
                        vm[(slot-1)*length(dict)+1:slot*length(dict),:]=vec_Mat
                        if(memCount!=0)
               for i=1:memoryLength
               vm[4*length(dict)+i,i]=1
               end
               end
               end
               
return vm
            end
#############################################################################################
                    lngth = length(dict)
                    o1_costmodel = compile(:o1_cost;mem_length=memoryLength) #Knet Model Compilation for o1_costmodel
                    o2_costmodel = compile(:o2_cost;mem_length=memoryLength)
                    #o2_costmodel = o1_costmodel
    r_costmodel = compile(:r_cost;dict_length=lngth)   #Knet Model Compilation for r_costmodel
    olr=0.0001 # sets initial lr for training o1_costmodel(for decaying lr)
                    setp(o1_costmodel; lr=olr)
                    setp(o2_costmodel; lr=0.0001)
    setp(r_costmodel; lr=0.0001) # sets initial lr for training r_costmodel
                    old=0 #for tracking 1 step older test performance
                    
                    
                    
                    
 ## Main Flow
 ############################################################################################
   #EPOCH Loop
    for(k=1:100)
       # println("epoch",k)
        trainmod=true # TRAIN/TEST switch
        global memCount=1    #Counter for memory location
        
        #initialisations for statistics
        trquestioncount=0
        trsum=0
        tr2sum=0
        tr_correct_response_count=0
        test_correct_response_count=0
        #################################
   #Train Data Loop     
    for i=1:size(data,1)
        sen=data[i,1]
        ans=data[i,2]
        
        if(data[i,3]!="")
            foo = split(data[i,3],' ',keep=true)
        clu=parse(Int,foo[1])
        clu2=parse(Int,foo[2])
        end
        
        newMem = I_module(sen[2],dict)
        
        if(sen[1]=="1") #new story
            memCount=1
            memAry = zeros(Float64,length(dict),memoryLength)

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
           
            y_memloc=zeros(size(memAry,2))
            y_memloc[clu]=1
        #1   trquestioncount+=1
            #if(memCount==15)
            #    println("15 : ",i)
            #    end
            y2_memloc=zeros(size(memAry,2))
            y2_memloc[clu2]=1
            memLoc,memloc2=O_module(memAry,newMem,y_memloc,y2_memloc,clu,trainmod,o1_costmodel,memCount)
             
           #println("Trprocess_memLoc:",memLoc)
           #println("Trprocess_clu:",clu)
            dictn = copy(dict)
            dictn[ans] = 1
        #1   if(clu==prob_vec2indice(memLoc))
        #1       trsum+=1
        #1   end
            
            y_answordloc = collect(values(dictn))
            #println(memAry[:,prob_vec2indice(memLoc)])
            #f_answer=R_module(memAry[:,prob_vec2indice(memLoc)], newMem, y_answordloc, dict, trainmod,r_costmodel)
            f_answer=R_module(memAry[:,clu],memAry[:,clu2], newMem, y_answordloc, dict, trainmod,r_costmodel)
            #println(i)
            #println("Answer : ",f_answer) 
            #println("RightAnswer : ",ans,".")
            #println("vector of R module response: ",get(r_costmodel,:r))
         #1      if(f_answer==ans)
         #1          tr_correct_response_count+=1
         #1      end
                   memCount+=1
        end#question
            

        end#data iterator(for)
           # if((trsum/trquestioncount)>0.5)
           #     setp(o1_costmodel, pdrop=0.77)
           # end
                
            #print(k,"\t",trsum/trquestioncount,"\t")
            #######################################
            trainmod=false
            ########### Training statistics
            for i=1:size(data,1)
        sen=data[i,1]
        ans=data[i,2]
        
        if(data[i,3]!="")
            foo = split(data[i,3],' ',keep=true)
        clu=parse(Int,foo[1])
            clu2=parse(Int,foo[2])
            end
        newMem = I_module(sen[2],dict)
        
        if(sen[1]=="1") #new story
            memCount=1
            memAry = zeros(Float64,length(dict),memoryLength)

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
           
            y_memloc=zeros(size(memAry,2))
            y_memloc[clu]=1
            trquestioncount+=1
             y2_memloc=zeros(size(memAry,2))
            y2_memloc[clu2]=1
            memLoc,memLoc2=O_module(memAry,newMem,y_memloc,y2_memloc,clu,trainmod,o1_costmodel,memCount)
             
           #
            dictn = copy(dict)
            dictn[ans] = 1
           if(clu==prob_vec2indice(memLoc))
               trsum+=1
           end
               if(clu2==prob_vec2indice(memLoc2))
               tr2sum+=1
           end
          # println("Tr:memLoc: ",memLoc)
          # println("Tr_clu: ",clu)
            y_answordloc = collect(values(dictn))
           
            #f_answer=R_module(memAry[:,prob_vec2indice(memLoc)], newMem, y_answordloc, dict, trainmod,r_costmodel)
            f_answer=R_module(memAry[:,clu],memAry[:,clu2], newMem, y_answordloc, dict, trainmod,r_costmodel)
           
               if(f_answer==ans)
                   tr_correct_response_count+=1
               end
                   memCount+=1
               end#question
               end#for
            ####
            ##############################################################
            ###Test
            trainmod=false
        memCount=1
               sum=0
               sum2=0
            questionquantity=0
 
    for i=1:size(test_data,1)
        sen=test_data[i,1]
        ans=test_data[i,2]
        
        if(test_data[i,3]!="")
            foo = split(test_data[i,3],' ',keep=true)
            clu=parse(Int,foo[1])
            
            clu2=parse(Int,foo[2])
            end
        newMem = I_module(sen[2],dict)
        
        if(sen[1]=="1") #new story
            memCount=1
            memAry = zeros(Float64,length(dict),memoryLength)

            end #clear mem,count
        if(!contains(sen[2],"?")) #not question
            G_module(memAry,newMem,memCount)
            memCount+=1
       
        else
            
            questionquantity+=1
            
            y_memloc=zeros(size(memAry,2))
            y_memloc[clu]=1

            y2_memloc=zeros(size(memAry,2))
            y2_memloc[clu2]=1
            memLoc,memLoc2=O_module(memAry,newMem,y_memloc,y2_memloc,clu,trainmod,o1_costmodel,memCount)
            
            #println("test:memLoc",memLoc)
            #println("clu",clu)
            dictn = copy(dict)
            dictn[ans] = 1
            if(clu==prob_vec2indice(memLoc))
                sum+=1
            #else
            #    println("thiiis")
            end
                if(clu2==prob_vec2indice(memLoc2))
                sum2+=1
            #else
            #    println("thiiis")
                end
            y_answordloc = collect(values(dictn))
            
            f_answer=R_module(memAry[:,clu],memAry[:,clu2], newMem, y_answordloc, dict, trainmod,r_costmodel)
            #f_answer=R_module(memAry[:,prob_vec2indice(memLoc)], newMem, y_answordloc, dict, trainmod,r_costmodel)
            #println("Answer : ",f_answer) 
            #println("RightAnswer : ",ans,".")
            #println("vector of R module response: ",get(r_costmodel,:r))
                if(f_answer==ans)
                   test_correct_response_count+=1
                end
                    memCount+=1
        end#question

        end#test_data iterator(for)
           
            println(k,"\t",trsum/trquestioncount,"\t",sum/questionquantity,"\t",tr2sum/trquestioncount,"\t",sum2/questionquantity,"\t R_Module training: ",tr_correct_response_count/trquestioncount,"\t test:",test_correct_response_count/questionquantity)
            #println(k,"\t",trsum/trquestioncount,"\t",sum/questionquantity,"\t",tr_correct_response_count/trquestioncount,"\t",test_correct_response_count/questionquantity)
            
            #For Decaying lr by o1_costmodel
            ###########################################
            #if(old>(sum/questionquantity))
            #    
            #    olr =0.8 * olr
            #    setp(o1_costmodel,lr=olr)
            #   
            #   setp(o1_costmodel, pdrop=0.50)
            #
            #end
            #    old=sum/questionquantity
            ###########################################
end#epochs

            ##


    ##
#end #module
