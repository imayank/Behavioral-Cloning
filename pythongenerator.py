def dataGenerator(data, batch_size,base_path_img):
    ids = data['ID'].values
    num = len(ids)
    #indices = np.arange(len(ids))
    np.random.seed(42)
    while True:
        #indices = shuffle(indices)
        ids = np.random.shuffle(ids)
        for offset in range(0,num,batch_size):
            batch = ids[offset:offset+batch_size]
            images = []
            target = []
            for batch_id in batch:
                img_path = data.loc[batch_id,'path']
                img_name = img_path.split('\\')[-1]
                new_path = base_path_img + img_name
                images.append(((mpimg.imread(new_path))/255)-0.5)
                target.append(data.loc[batch_id,'target'])
                
            images = np.array(images)
            target = np.array(target)
            
            yield images, target
                
        
        
    
    