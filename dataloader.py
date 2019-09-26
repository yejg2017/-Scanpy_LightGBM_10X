import os
import scanpy as sc
import numpy as np
import pickle
import pandas as pd
import argparse



class loader(object):
    def __init__(self,root,metadata_dir,gene_dir,save_dir="./log"):
        self.root=root
        self.metadata_dir=metadata_dir
        self.gene_dir=gene_dir
        
        self._datasets()
        self.load_meta_gene()

        self.save_dir=save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def load_meta_gene(self):
        metadata=pd.read_csv(self.metadata_dir,sep=",")
        genes=pd.read_csv(self.gene_dir)

        clusters=metadata[["celltype"]].values
        self.clusters=[clusters[i][0] for i in range(len(clusters))]
        self.cells=[str(x) for x in metadata.index]

        genes=genes.values
        self.genes=[genes[i][0] for i in range(len(genes))]
        self.cell2cluster={cell:ident for cell,ident in zip(self.cells,self.clusters)}
    

    def _datasets(self):
        file=open("sample0.txt","r")
        datasets=[]
        for line in file.readlines():
            dataset=line.split("\t")[0]
            datasets.append(dataset)
        file.close()
        self.datasets=datasets
    
    def load_one_10X(self,dataset):
        splits=dataset.split("-")
        prefix=splits[0]+"-"+splits[1]
        path=os.path.join(self.root,dataset,"outs/filtered_feature_bc_matrix.h5")
        adata=sc.read_10x_h5(path)
        #adata.var_names_make_unique()
        cells=[prefix+"_"+str(x) for x in adata.obs_names]
        #genes=[str(x) for x in adata.var_names]
        adata.obs_names=cells
        return adata
    
    def load_all(self):
        print("Initial...")
        print("Loading from : No.1 dataset -- {}".format(self.datasets[0]))
        adata=self.load_one_10X(self.datasets[0])
        for i,dataset in enumerate(self.datasets[1:]):
            print("Loading from : No.{} dataset -- {}".format(str(i+2),dataset))
            data=self.load_one_10X(dataset)
            adata=adata.concatenate(data,index_unique=None)

        self.adata=adata
        print("There are {} cells,{} genes".format(self.adata.n_obs,self.adata.n_vars))
        del adata

        cells=[str(cell) for cell in self.adata.obs_names]
        genes=[str(gene) for gene in self.adata.var_names]
        cells_index={cell:idx for idx,cell in enumerate(cells)}
        
        data={"raw":{"adata":self.adata,
                     "cell2idx":cells_index,
                     "genes":genes
                     },
              "reference":{"genes":self.genes,
                           "cell2cluster":self.cell2cluster
                           }
              }
        #print("save data")
        #self.save_file=os.path.join(self.save_dir,"loader.pkl")
        #with open(self.save_file,"wb") as fp:
        #     pickle.dump(data,fp)
        #fp.close()
        self.data=data
        #print("Save Done")

    def _cell2cluster(self):
        all_cells=[str(cell) for cell in self.adata.obs_names]
        cells_index={cell:idx for idx,cell in enumerate(all_cells)}
        reference_cells=self.cells

        all_genes=[str(gene) for gene in self.adata.var_names]
        genes_index={gene:idx for idx,gene in enumerate(all_genes)}
        reference_genes=self.genes

        subset_cells=list(set(all_cells).intersection(reference_cells))
        subset_genes=list(set(all_genes).intersection(reference_genes))

        cell_idx=[cells_index[cell] for cell in subset_cells]
        gene_idx=[genes_index[gene] for gene in subset_genes]
        
        cell_cluster_prediction=[self.cell2cluster[cell] for cell in subset_cells]

        x=self.adata[cell_idx,:]
        x=x[:,gene_idx]

        self.data["save"]={"adata":x,
                                 "cluster":cell_cluster_prediction
                                 }
        self.save_file=os.path.join(self.save_dir,"loader.pkl")
        print("save data")
        with open(self.save_file,"wb") as fp:
             pickle.dump(self.data,fp)
        fp.close()
        print("Save Done")
    
    def to_array(self):
        fp=open(self.save_file,"rb")
        data=pickle.load(fp)

        adata=data["save"]["adata"]

        genes=adata.var_names
        cells=adata.obs_names
        array=adata.X.toarray()

        data={"array":array,
              "genes":genes,
              "cells":cells,
              "cluster":data["save"]["cluster"]
                }
        filename=os.path.join(os.path.dirname(self.save_file),"array.pkl")
        with open(filename,"wb") as fp:
            pickle.dump(data,fp,protocol=4)
        fp.close()
        print("write array done")
    def _process(self,data):
        adata=data.copy()
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        sc.pp.log1p(adata)

        sc.pp.scale(adata, max_value=10)
        return adata



    def __repr__(self):
        fmt_str = "Object : " + self.__class__.__name__ + "\n"
        fmt_str += "Dataset root :{}\n".format(self.root)
        fmt_str += "Dataset meta path : {}\n".format(self.metadata_dir)
        fmt_str += "Dataset gene path : {}\n".format(self.gene_dir)
        fmt_str += "Number of datasets : {}\n".format(len(self.datasets))
        return fmt_str




if __name__=="__main__":
   genedir="/home/ye/Work/R/10X/Human/VDJ/VKH/pretrain/all_ratio_1.0_seed_6666_resolution_0.8_sample0.txt_time_2019-09-12/model/genes.csv"
   root="/Data/zoc/result/10X-count/PBMC/10X-VDJ-human/5RNA"
   metadir="/home/ye/Work/R/10X/Human/VDJ/VKH/pretrain/all_ratio_1.0_seed_6666_resolution_0.8_sample0.txt_time_2019-09-12/model/metadata.csv"
   dataset=loader(root,metadir,genedir)
   print(dataset)
   dataset.load_all()
   dataset._cell2cluster()
   dataset.to_array()

