import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
import torch
import pickle
from torch_geometric.utils import dense_to_sparse

#TODO aggiungi dataset syntie e aggiungi le maschere e altre cose a tutti i dataset

def get_dataset(dataset_name: str = None, test_size: float = 0.2)->Data:
    """_summary_

    Args:
        dataset_name (str, optional): _description_. Defaults to None.

    Returns:
        Data: _description_
    """
    if dataset_name in ["cora", "pubmed", "citeseer"]:
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root="data", name=dataset_name) [0]      
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])]) 
        
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]

        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "karate":
        from torch_geometric.datasets import KarateClub
        
        dataset = KarateClub()[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))

        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "twitch":
        from torch_geometric.datasets import Twitch

        dataset = Twitch(root="data", name="EN")[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]  
        
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "actor":
        from torch_geometric.datasets import Actor
        

        dataset = Actor(root="data")[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        train_index, test_index = train_test_split(ids, test_size=0.03, random_state=random.randint(0, 100))
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        from torch_geometric.datasets import WebKB
        

        dataset = WebKB(root="data", name=dataset_name)[0]  
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=dataset.y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)   
     
    elif dataset_name in ["Wiki", "BlogCatalog", "Facebook", "PPI"]:
        from torch_geometric.datasets import AttributedGraphDataset

        dataset = AttributedGraphDataset(root="data", name=dataset_name)[0]
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1)
        y = dataset.y if dataset_name != "Facebook" else torch.argmax(dataset.y, dim=1)
        
        ids = torch.arange(start=0, end=dataset.x.shape[0]-1, step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.x, dim=0)[0]
        max_range = torch.max(dataset.x, dim=0)[0]
        
        discrete_mask = torch.Tensor([1 for i in range(dataset.x.shape[1])])
        
        return Data(x=dataset.x, edge_index=dataset.edge_index, y=y, train_mask=train_index, test_mask=test_index, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)           
        
    elif "syn" in dataset_name:
        with open(f"data/{dataset_name}.pickle","rb") as f:
            data = pickle.load(f)

        adj = torch.Tensor(data["adj"]).squeeze()  
        features = torch.Tensor(data["feat"]).squeeze()
        labels = torch.tensor(data["labels"]).squeeze()
        idx_train = data["train_idx"]
        idx_test = data["test_idx"]
        edge_index = dense_to_sparse(adj)   

        train_index, test_index = train_test_split(idx_train + idx_test, test_size=test_size, random_state=random.randint(0, 100))  
        
        return Data(x=features, edge_index=edge_index[0], y=labels, train_mask=idx_train, test_mask=idx_test)
    
    elif dataset_name == "AIDS":

        class AIDS(InMemoryDataset):
            def __init__(self, root, transform=None, pre_transform=None):
                super(AIDS, self).__init__(root, transform, pre_transform)
                self.data, self.slices = torch.load(self.processed_paths[0])
                self.discrete_mask = torch.Tensor([1, 1, 0, 0])

            @property
            def raw_file_names(self):
                return ["AIDS_A.txt", "AIDS_graph_indicator.txt", "AIDS_graph_labels.txt", "AIDS_node_labels.txt", "AIDS_node_attributes.txt"]

            @property
            def processed_file_names(self):
                return ["data.pt"]

            def download(self):
                pass

            def process(self):
                # Read data into huge `Data` list.
                data_list = []

                # Read files
                edge_index = pd.read_csv(os.path.join(self.raw_dir, "AIDS_A.txt"), sep=",", header=None).values.T
                graph_indicator = pd.read_csv(os.path.join(self.raw_dir, "AIDS_graph_indicator.txt"), sep=",", header=None).values.flatten()
                graph_labels = pd.read_csv(os.path.join(self.raw_dir, "AIDS_graph_labels.txt"), sep=",", header=None).values.flatten()
                node_labels = pd.read_csv(os.path.join(self.raw_dir, "AIDS_node_labels.txt"), sep=",", header=None).values.flatten()
                node_attributes = pd.read_csv(os.path.join(self.raw_dir, "AIDS_node_attributes.txt"), sep=",", header=None).values

                # Process data
                for graph_id in range(1, graph_indicator.max() + 1):
                    node_mask = graph_indicator == graph_id
                    nodes = torch.tensor(node_mask.nonzero()[0].flatten(), dtype=torch.long)
                    x = torch.tensor(node_attributes[node_mask], dtype=torch.float)
                    y = torch.tensor(node_labels[node_mask], dtype=torch.long)

                    edge_mask = (graph_indicator[edge_index[0] - 1] == graph_id) & (graph_indicator[edge_index[1] - 1] == graph_id)
                    edges = torch.tensor(edge_index[:, edge_mask] - 1, dtype=torch.long)

                    data = Data(x=x, edge_index=edges, y=y)
                    data_list.append(data)

                data, slices = self.collate(data_list)
                print('slices', slices)
                print('data', data)
                torch.save((data, slices), self.processed_paths[0])
        
        dataset = AIDS(root="data/AIDS")
        print('feat_mat', dataset.data.x)
        print('labels', dataset.data.y)
        ids = torch.arange(start=0, end=len(dataset.data.x), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]
        print(f"Stats:\nFeatures:{dataset.data.x.shape[1]}\nNodes:{dataset.data.x.shape[0]}\nEdges:{dataset.data.edge_index.shape[1]}\nClasses:{dataset.data.y.max().item()}\n")
        
        final = Data(x=dataset.data.x, edge_index=dataset.data.edge_index, y=dataset.data.y, train_mask=train_index, test_mask=test_index, discrete_mask=dataset.discrete_mask, min_range=min_range, max_range=max_range)
        print(final.train_mask)
        return final
    


    elif dataset_name == "NodeCoderDataset":
        # Define your custom dataset class inline (similar to the AIDS branch)
        import os
        import torch
        import pandas as pd
        from torch_geometric.data import Data, InMemoryDataset, Batch
        from sklearn.model_selection import train_test_split
        torch.set_printoptions(sci_mode=False, precision=3)

        class NodeCoderDataset(InMemoryDataset):
            def __init__(self, root, split="train", transform=None, pre_transform=None):
                self.split = split  # differentiate train/val if needed
                super(NodeCoderDataset, self).__init__(root, transform, pre_transform)
                # Load the processed (batched) Data object and discrete mask together.
                self.data, self.discrete_mask = torch.load(self.processed_paths[0], map_location='cpu')

            @property
            def raw_file_names(self):
                return [
                    "train_1_nodes_ProteinID.csv",
                    "train_1_features.csv",
                    "train_1_edges.csv",
                    "train_1_edge_features.csv",
                    "train_1_target.csv"
                ]

            @property
            def processed_file_names(self):
                return ["data.pt"]

            def download(self):
                pass

            def process(self):
                raw_dir = self.raw_dir

                # 1. Read nodes and group by protein id.
                nodes_path = os.path.join(raw_dir, "train_1_nodes_ProteinID.csv")
                nodes_df = pd.read_csv(nodes_path)
                nodes_df.columns = nodes_df.columns.str.strip()
                protein_ids = sorted(nodes_df["protein_id"].unique().tolist())

                # 2. Process features.
                features_path = os.path.join(raw_dir, "train_1_features.csv")
                features_df = pd.read_csv(features_path)
                features_df.columns = features_df.columns.str.strip()
                pivoted_features = features_df.pivot_table(
                    index="node_id",
                    columns="feature_id",
                    values="value",
                    aggfunc='first'
                ).fillna(0)
                
                # Create a discrete mask: 1 if the feature values are all integers, 0 otherwise.
                self.discrete_mask = torch.tensor([
                    1 if (pivoted_features[col] % 1 == 0).all() else 0
                    for col in pivoted_features.columns
                ], dtype=torch.float)

                # 3. Process targets.
                target_path = os.path.join(raw_dir, "train_1_target.csv")
                target_df = pd.read_csv(target_path)
                target_df.columns = target_df.columns.str.strip()
                if "node_id" not in target_df.columns:
                    targets = target_df.copy()
                    targets.index = nodes_df["node_id"].tolist()
                else:
                    targets = target_df.set_index("node_id")

                # Convert targets to numeric (ensuring correct sum computation)
                targets = targets.apply(pd.to_numeric, errors='coerce')
                # Drop target columns that are all zeros.
                valid_cols = targets.columns[(targets.sum(axis=0) > 0)]
                targets = targets[valid_cols]
                print('Processed targets:')
                print(targets.head())

                # 4. Process edges and edge features.
                edges_path = os.path.join(raw_dir, "train_1_edges.csv")
                edges_df = pd.read_csv(edges_path)
                edges_df.columns = edges_df.columns.str.strip()
                
                edge_features_path = os.path.join(raw_dir, "train_1_edge_features.csv")
                edge_features_df = pd.read_csv(edge_features_path)
                edge_features_df.columns = edge_features_df.columns.str.strip()

                # 5. Build individual Data objects (each with local indices).
                data_list = []
                for protein in protein_ids:
                    sub_nodes = nodes_df[nodes_df["protein_id"] == protein]["node_id"].tolist()
                    local_map = {nid: i for i, nid in enumerate(sub_nodes)}
                    
                    x_sub = torch.tensor(
                        pivoted_features.reindex(sub_nodes, fill_value=0).values,
                        dtype=torch.float
                    )
                    y_sub = torch.tensor(
                        targets.reindex(sub_nodes, fill_value=0).values,
                        dtype=torch.float
                    )
                    
                    sub_edges = []
                    sub_edge_feats = []
                    edge_feat_map = {}
                    for _, row in edge_features_df.iterrows():
                        id1, id2 = row["id1"], row["id2"]
                        if id1 in local_map and id2 in local_map:
                            feats = row.drop(labels=["id1", "id2"]).values.tolist()
                            if len(feats) != 3:
                                raise ValueError("Expected 3 edge features")
                            edge_feat_map[(local_map[id1], local_map[id2])] = feats

                    for _, row in edges_df.iterrows():
                        id1, id2 = row["id1"], row["id2"]
                        if id1 in local_map and id2 in local_map:
                            src = local_map[id1]
                            dst = local_map[id2]
                            sub_edges.append([src, dst])
                            if (src, dst) in edge_feat_map:
                                sub_edge_feats.append(edge_feat_map[(src, dst)])
                            else:
                                raise ValueError(f"Missing edge features for edge ({src},{dst})")
                                
                    if len(sub_edges) > 0:
                        edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
                        edge_attr = torch.tensor(sub_edge_feats, dtype=torch.float)
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                        edge_attr = torch.empty((0, 3), dtype=torch.float)
                    
                    data = Data(x=x_sub, y=y_sub, edge_index=edge_index, edge_attr=edge_attr)
                    data.protein_id = protein
                    data_list.append(data)
                
                # Batch the individual Data objects into a single Data object with continuous global node indices.
                batched_data = Batch.from_data_list(data_list)
                torch.save((batched_data, self.discrete_mask), self.processed_paths[0])

            def get_final_data(self):
                # Create train/test splits over nodes.
                ids = torch.arange(0, self.data.x.shape[0]).tolist()
                train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
                min_range = torch.min(self.data.x, dim=0)[0]
                max_range = torch.max(self.data.x, dim=0)[0]
                
                final_data = Data(
                    x=self.data.x,
                    edge_index=self.data.edge_index,
                    y=self.data.y,
                    train_mask=train_index,
                    test_mask=test_index,
                    min_range=min_range,
                    max_range=max_range
                )
                final_data.discrete_mask = self.discrete_mask
                return final_data

        # Instantiate and process the dataset.
        dataset = NodeCoderDataset(root="data/NodeCoderDataset")
        final_data = dataset.get_final_data()
        print(final_data)
        print('train_mask_shape', final_data.train_mask)
        print('shape_di_y', final_data.y.shape)
        return final_data


    elif dataset_name == "enzymes":
        class ENZYMES(InMemoryDataset):
            def __init__(self, root, transform=None, pre_transform=None):
                super(ENZYMES, self).__init__(root, transform, pre_transform)
                self.data, self.slices = torch.load(self.processed_paths[0])
                self.discrete_mask = torch.Tensor([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            @property
            def raw_file_names(self):
                return ["ENZYMES_A.txt", "ENZYMES_graph_indicator.txt", "ENZYMES_graph_labels.txt", "ENZYMES_node_labels.txt", "ENZYMES_node_attributes.txt"]

            @property
            def processed_file_names(self):
                return ["data.pt"]

            def download(self):
                pass

            def process(self):
            # Read data into huge `Data` list.
                data_list = []

                # Read files
                edge_index = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_A.txt"), sep=",", header=None).values.T
                graph_indicator = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_graph_indicator.txt"), sep=",", header=None).values.flatten()
                graph_labels = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_graph_labels.txt"), sep=",", header=None).values.flatten()
                node_labels = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_node_labels.txt"), sep=",", header=None).values.flatten()
                node_attributes = pd.read_csv(os.path.join(self.raw_dir, "ENZYMES_node_attributes.txt"), sep=",", header=None).values

                # Process data
                for graph_id in range(1, graph_indicator.max() + 1):
                    node_mask = graph_indicator == graph_id
                    nodes = torch.tensor(node_mask.nonzero()[0].flatten(), dtype=torch.long)
                    x = torch.tensor(node_attributes[node_mask], dtype=torch.float)
                    y = torch.tensor(node_labels[node_mask], dtype=torch.long) - 1

                    edge_mask = (graph_indicator[edge_index[0] - 1] == graph_id) & (graph_indicator[edge_index[1] - 1] == graph_id)
                    edges = torch.tensor(edge_index[:, edge_mask] - 1, dtype=torch.long)

                    data = Data(x=x, edge_index=edges, y=y)
                    data_list.append(data)

                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])

        dataset = ENZYMES(root="data/ENZYMES")
        ids = torch.arange(start=0, end=len(dataset.data.x), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]
        
        print(f"Stats:\nFeatures:{dataset.data.x.shape[1]}\nNodes:{dataset.data.x.shape[0]}\nEdges:{dataset.data.edge_index.shape[1]}\nClasses:{dataset.data.y.max().item()}\n")
       
        return Data(x=dataset.data.x, edge_index=dataset.data.edge_index, y=dataset.data.y, train_mask=train_index, test_mask=test_index,  discrete_mask=dataset.discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "protein":
        
        class Proteins(InMemoryDataset):
            def __init__(self, root, transform=None, pre_transform=None):
                super(Proteins, self).__init__(root, transform, pre_transform)
                self.discrete_mask = torch.Tensor([1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

                self.data, self.slices = torch.load(self.processed_paths[0])

            @property
            def raw_file_names(self):
                return ["PROTEINS_full_A.txt", "PROTEINS_full_graph_indicator.txt", "PROTEINS_full_graph_labels.txt", "PROTEINS_full_node_labels.txt", "PROTEINS_full_node_attributes.txt"]

            @property
            def processed_file_names(self):
                return ["data.pt"]

            def download(self):
                pass

            def process(self):
                data_list = []

                edge_index = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_A.txt"), sep=",", header=None).values.T
                graph_indicator = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_graph_indicator.txt"), sep=",", header=None).values.flatten()
                graph_labels = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_graph_labels.txt"), sep=",", header=None).values.flatten()
                node_labels = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_node_labels.txt"), sep=",", header=None).values.flatten()
                node_attributes = pd.read_csv(os.path.join(self.raw_dir, "PROTEINS_full_node_attributes.txt"), sep=",", header=None).values

                for graph_id in range(1, graph_indicator.max() + 1):
                    node_mask = graph_indicator == graph_id
                    nodes = torch.tensor(node_mask.nonzero()[0].flatten(), dtype=torch.long)
                    x = torch.tensor(node_attributes[node_mask], dtype=torch.float)
                    y = torch.tensor(node_labels[node_mask], dtype=torch.long)

                    edge_mask = (graph_indicator[edge_index[0] - 1] == graph_id) & (graph_indicator[edge_index[1] - 1] == graph_id)
                    edges = torch.tensor(edge_index[:, edge_mask] - 1, dtype=torch.long)

                    data = Data(x=x, edge_index=edges, y=y)
                    data_list.append(data)

                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[0])

        dataset = Proteins(root="data/PROTEINS_full")
        ids = torch.arange(start=0, end=len(dataset.data.x), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.005, random_state=random.randint(0, 100))
        
        print(f"Stats:\nFeatures:{dataset.data.x.shape[1]}\nNodes:{dataset.data.x.shape[0]}\nEdges:{dataset.data.edge_index.shape[1]}\nClasses:{dataset.data.y.max().item()}\n")
        
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]

        return Data(x=dataset.data.x, edge_index=dataset.data.edge_index, y=dataset.data.y, train_mask=train_index, test_mask=test_index,  discrete_mask=dataset.discrete_mask, min_range=min_range, max_range=max_range)

    elif dataset_name == "AIDS-G":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([1, 1, 0, 0] + [1] * 38)
        dataset = TUDataset(root="data/aids", name="AIDS", use_node_attr=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        print('aids_g', dataset)
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=0.2, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "ENZYMES-G":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([0, 0, 0, 0, 0, 0] + [1] * 15)
        dataset = TUDataset(root="data", name="ENZYMES", use_node_attr=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)
    
    elif dataset_name == "PROTEINS-G":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([1, 1, 1, 0, 1, 0, 0, 0, 0] + [1] * 12 + [0] * 8 + [1, 1, 1])
        dataset = TUDataset(root="data", name="PROTEINS_full", use_node_attr=True, force_reload=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)                
    
    elif dataset_name == "COIL-DEL":
        
        from torch_geometric.datasets import TUDataset
        
        discrete_mask = torch.Tensor([1, 1])
        dataset = TUDataset(root="data", name="COIL-DEL", use_node_attr=True, force_reload=True)
        min_range = torch.min(dataset.data.x, dim=0)[0]
        max_range = torch.max(dataset.data.x, dim=0)[0]  
        
        ids = torch.arange(start=0, end=len(dataset), step=1).tolist()
        train_index, test_index = train_test_split(ids, test_size=test_size, random_state=random.randint(0, 100))
        
        train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        
        train_mask[train_index] = True
        test_mask[test_index] = True
              
        return Data(dataset=dataset, train_mask=train_mask, test_mask=test_mask, discrete_mask=discrete_mask, min_range=min_range, max_range=max_range)                
        
    else:
        raise Exception("Choose a valid dataset!")
    
    

if __name__ == "__main__":
    
    
    data = get_dataset("PROTEINS_G")
    
    print(data)