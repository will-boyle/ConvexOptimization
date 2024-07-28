This library can be used to solve constrainted optimization problems. 

What kind of problems can it solve?


1) Linear Programs (pm just works)

min cx
 
st Ax <= b
    x >= 0



2) Integer Programs (Think integer program might have bug in it.)
   
min cx

st Ax <= b
    x >= 0
    x_i is an integer for all i
