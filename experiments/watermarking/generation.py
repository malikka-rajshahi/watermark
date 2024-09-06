import torch

def generate(model,prompts,vocab_size,n,m,seeds,key_func,sampler,random_offset=True,token_embeddings=None,beta=1):
    print("made it to generate")
    batch_size = len(prompts)

    generator = torch.Generator()
    xis,pis = [],[]
    print("populating xis and pis")
    print(seeds)
    for seed in seeds:
        generator.manual_seed(int(seed))
        if token_embeddings is None:
            out_file = f'output-{m}-gumbel.txt'
            file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
            file.write('made it to generate\n')
            file.close()
            xi,pi = key_func(generator,n,vocab_size)
        else:
            out_file = f'output-{m}-gumbel_mod.txt'
            file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
            file.write('made it to generate\n')
            file.close()
            xi,pi,token_embeddings,beta = key_func(generator,n,vocab_size,token_embeddings,out_file,beta=beta)
        print("set xi and pi")
        file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
        file.write("set xi and pi\n")
        file.close()
        xis.append(xi.unsqueeze(0))
        pis.append(pi.unsqueeze(0))

    xis = torch.vstack(xis)
    pis = torch.vstack(pis)
    pis = pis.to(dtype=torch.int64)
    print("xis and pis populated")
    file = open(f'/scratch/projects/hegdelab/mr6177/watermark/{out_file}', 'a')
    file.write("xis and pis populated\n")
    file.close()

    # deliberately not controlling this randomness with the generator
    if random_offset:
        offset = torch.randint(n,size=(batch_size,))
    else:
        offset = torch.zeros(size=(batch_size,),dtype=torch.int64)
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1).cpu()
        tokens = sampler(probs, pis, xis[torch.arange(batch_size),(offset.squeeze()+i)%n]).to(model.device)
        inputs = torch.cat([inputs, tokens], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()

# generate unwatermarked completions of token length m given list of prompts
def generate_rnd(prompts,m,model):
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:,-1], dim=-1)
        
        tokens = torch.multinomial(probs,1)
        inputs = torch.cat([inputs, tokens], dim=1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
    
    return inputs.detach().cpu()
