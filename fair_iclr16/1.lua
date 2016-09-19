require 'torch'
require 'nn'
require 'nngraph'
require 'os'

p = 6
lr = 0.01
b = 5


--model = nn.gModule({})

context = torch.Tensor(2, 5):fill(2)
dict = nn.ParallelTable()
for i = 1, b do dict:add(nn.LookupTable(10, 6)) end
o1 = nn.SplitTable(1, 1):forward(context)
o2 = dict:forward(o1)
print(o2)
o3 = nn.JoinTable(1,1):forward(o2)
print(o3)
os.exit(0)

query = torch.Tensor{2}
q_look = nn.LookupTable(2, 6)
q_out = q_look:forward(query)
print(q_out)

pred = nn.MM(false, true):forward({o3, q_out})
print(pred)

_, m_bar = pred:max(1)
m_o1 = nil
rel_mem = torch.Tensor{1,5} -- all answer memories (put a map)
if (#rel_mem)[1] == 1 then
  m_01 = rel_mem[1]
else
  max = -1
  for j = 1, (#rel_mem)[1] do
  	if max==-1 or pred[rel_mem[max]][1] > pred[rel_mem[j]][1] then
  		max = j
  		print(max)
  	end
  end
  assert(max ~= -1)
  m_o1 = rel_mem[max]
end
print(m_o1)

os.exit(0)
-- context[1][1]:fill(1)
-- context[1][2]:fill(2)
-- context[1][3]:fill(3)
-- context[1][4]:fill(0)
-- context[2][1]:fill(1)
-- context[2][2]:fill(2)
-- context[2][3]:fill(3)
-- context[2][4]:fill(0)

query = torch.Tensor(2, 4)
query[1]:fill(1)
query[2]:fill(2)

answer = torch.Tensor(2, 1)
answer[1] = 1
answer[2] = 2

A = nn.Linear(4, 6)
A_c = A:clone('weight','bias','gradWeight','gradBias')

model:forward()