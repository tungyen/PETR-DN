#---hyperparameter-------------------------------------------------------------
scalar=5
label_noise_scale=0.2 
box_noise_scale=0.4 
num_patterns=0


# Generate a random embedding matrix using PyTorch
embedding_matrix = torch.rand(92, 255) * 2 - 1  # Random values between -1 and 1
random_tensor = torch.rand(300, 4)
random_tensor = random_tensor.to('cuda')
parameter = nn.Parameter(random_tensor)
embedweight=parameter
batch_size=3
training=True
num_queries=300
num_classes=91
hidden_dim=256
label_enc=nn.Embedding(92,255)
#--------------------------------------------------------

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def prepare_for_dn(dn_args, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    prepare for dn components in forward function
    Args:
        dn_args: (targets, args.scalar, args.label_noise_scale,
                                                             args.box_noise_scale, args.num_patterns) from engine input
        embedweight: positional queries as anchor
        training: whether it is training or inference
        num_queries: number of queries
        num_classes: number of classes
        hidden_dim: transformer hidden dimenstion
        label_enc: label encoding embedding

    Returns: input_query_label, input_query_bbox, attn_mask, mask_dict
    """
    #training 才要noise
    if training:
        targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
    else:
        num_patterns = dn_args

    if num_patterns == 0:
        num_patterns = 1
        
    # (300, 1), 300個0的向量
    indicator0 = torch.zeros([num_queries * num_patterns, 1]).cuda()
    
    # (300, 255) 
    tgt = label_enc(torch.tensor(num_classes).cuda()).repeat(num_queries * num_patterns, 1)  #num_classes=91
    
    # (300, 256)
    tgt = torch.cat([tgt, indicator0], dim=1)
    
    # (300, 4)
    refpoint_emb = embedweight.repeat(num_patterns, 1)
    
    #ex:有三個image data :第一張圖有2個GT, 第二張圖有4個GT, 第三張圖有3個GT,
    
    if training:
        #都是1的tensor, 是一個list, 數量是bs的大小, 裡面的tensor的大小是每個image中的target數量 
        # ex:第一張圖有2個GT:[1, 1], 第二張圖有4個GT:[1, 1, 1, 1], 第三張圖有3個GT:[1, 1, 1]
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        
        #那些target的index ex:第一張圖有2個GT:[0, 1], 第二張圖有4個GT:[0,1,2,3], 第三張圖有3個GT:[0, 1, 2]
        know_idx = [torch.nonzero(t) for t in known]
        
        #每張圖上有幾個GT ex:第一張圖有2個GT:[2], 第二張圖有4個GT:[4], 第三張圖有3個GT:[3]
        known_num = [sum(k) for k in known]
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     scalar=scalar//int(max(known_num))

        # can be modified to selectively denosie some label or boxes; also known label prediction
        #一共有幾個GT 用1表示  ex:總共有9個 [1,1,1,1,1,1,1,1,1]
        unmask_bbox = unmask_label = torch.cat(known)
        
        #取出所有GT的label(哪一類)
        #隨便舉例:[7,1,38,4,6,32,2,1,1]
        labels = torch.cat([t['labels'] for t in targets])
        
        #取出所有GT的box(xywh)
        #(9, 6)   九個gt每個xyzwhl
        boxes = torch.cat([t['boxes'] for t in targets])
        
        # 標示屬於那些圖片 
        #[0,0,1,1,1,1,2,2,2]
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        
        #返回一個二維tensor
        #(9,1)
        #[0,1,2,3,4,5,6,7,8]
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        
        #(N,1)-->(N,) 拉平
        known_indice = known_indice.view(-1)

        # add noise
        #假設一共有"scalar'組, scalar=5
        
        #一共有5組  重複5次 拉平
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        
        #gt label 重複5次 拉平
        known_labels = labels.repeat(scalar, 1).view(-1)
        
        #gt所屬的image_id  重複5次 拉平
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        
        #gt bbox 重複5次 拉平
        known_bboxs = boxes.repeat(scalar, 1)
        
        #clone
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        
        #接著處理
        #1. label noise
        #2. box noise

        # 1. noise on the label
        if label_noise_scale > 0:
            #隨機值0-1內 
            #known label=9個gtx5組=45
            #隨機生成45個0-1之間的值
            p = torch.rand_like(known_labels_expaned.float()) #ex:[0.1234, 0.4553, 0.5674, 0.1134......]
            
            #那些<0.2的index   ex:[0,3.....]
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
            
            #給那些被選擇的GT 一個隨意的label id
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            
            #把新的值塞回去
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
            
        
        # 2. noise on the box
        if box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand)  #[45,6]  diff=[[0,0,0,0,0,0],[0,0,0,0,0,0].....]
            
            #把w h l的一半放到中心座標的地方
            diff[:, :3] = known_bbox_expand[:, 3:] / 2
            
            # w h l 還是 w h l
            diff[:, 3:] = known_bbox_expand[:, 3:]
            #diff=[w, h, l, w/2, h/2, l/2]
            
            #論文中計算xywh加噪音的計算方式
            #Δx<λ1w/2
            #Δy<λ1h/2
            #Δz<λ1l/2
            #[(1-λ2)w, (1+λ2)w]
            #[(1-λ2)h, (1+λ2)h]
            #[(1-λ2)l, (1+λ2)l]
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                           diff).cuda() * box_noise_scale
            
            #裁剪，防止溢出
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)
            
            
        #m是之前那個翻轉過的label, (45,)
        m = known_labels_expaned.long().to('cuda')
        
        #(45, 255)將label_tensor傳入label_enc做embedding---->CLASS LABEL EMBEDDING
        input_label_embed = label_enc(m)
        
        # add dn part indicator
        #(45,1)全是1的tensor
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
        
        #(45, 256) tgt相關的是補0, tgt是給正常匹配的使用, 這裡補的是1
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        
        #對座標取反函數(N, 4) 對應於特徵圖上的座標
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        
        #bs中含有最多target的數量， ex:第二張圖有4個GT---> 4
        single_pad = int(max(known_num))
        
        #(4x5=20)
        pad_size = int(single_pad * scalar)
        
        #(20,256)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        
        #(20,4)
        padding_bbox = torch.zeros(pad_size, 4).cuda()
        
        #拼在前面的是去躁部分(padding_label)，在後面的是正常匹配(tgt)的部分
        #與正常的，需要經過匈牙利匹配的那一部分query拼在一起
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        
        #refpoint_emb是正常的 給到300個預測使用的部分
        input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)

        # map in order
        #開一個空的
        map_known_indice = torch.tensor([]).to('cuda')
        
        if len(known_num):
            #各個image的GT合併在一起 ex:[0, 1, 0,1,2,3, 0, 1, 2]
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  
            
            #對各個group加上偏移量，這個偏移量是這些batch中最大的gt數量
            # ex:第一組不加偏移量 第2組加4 第3組加8
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            
        #known_bid是gt所屬的image_id
        if len(known_bid):
            #known_bid(5N,) map_known_indice(5N, ) 替換對應的embed
            #input_query_label為[bs, 300+5N, 256] 其實也就是替換了各個image的前5N個
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            
            
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed
            
            
        #tgt_size 為300+5N ex:300+20
        tgt_size = pad_size + num_queries * num_patterns
        
        #初始值都是false, 0(false)表示可以看見 1(true)表示被mask看不見 (300+5N, 300+5N) 初始先認為他都是可以看見的
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        
        #第一個pad_size:表示是這些group之後，就是正常需要進行匹配的那300個，在那300個裡面 不能看見前面的那些group
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        
        # reconstruct cannot see each other
        for i in range(scalar):
            #第一組
            if i == 0:
                #single_pad是bs中擁有最多gt的gt數量
                #看不見他後面所有的group
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            
            #最後一組    
            if i == scalar - 1:
                #看不見他前面所有的group
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            #中間組
            else:
                #看不見他後面的group
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                
        mask_dict = {
            #變成long類型
            'known_indice': torch.as_tensor(known_indice).long(),
            #標示gt屬於哪一張image
            'batch_idx': torch.as_tensor(batch_idx).long(),
            
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'pad_size': pad_size
        }
    else:  
        # 推理模式的時候沒有噪聲
        # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    input_query_label = input_query_label.transpose(0, 1)
    input_query_bbox = input_query_bbox.transpose(0, 1)

    return input_query_label, input_query_bbox, attn_mask, mask_dict