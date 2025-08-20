class FeatureExtractor:
    """Standardized feature extraction for all CBM methods"""
    
    def extract(self, dataset, backbone_config):
        backbone = self._load_backbone(backbone_config)
        
        features = []
        labels = []
        
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                # Standardized feature extraction
                batch_features = self._extract_batch_features(backbone, batch_x)
                features.append(batch_features)
                labels.append(batch_y)
        
        return {
            'features': torch.cat(features, dim=0),
            'labels': torch.cat(labels, dim=0)
        }
    
    def _extract_batch_features(self, backbone, x):
        """Consistent feature extraction across methods"""
        if "clip" in str(backbone):
            features = backbone.encode_image(x)
        elif hasattr(backbone, 'features'):  # CUB models
            features = backbone.features(x)
        else:  # Standard models
            features = backbone(x)
        
        # Flatten if needed
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
        
        return features