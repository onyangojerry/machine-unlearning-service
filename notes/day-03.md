read deletion request
look up each record in shard manifest
collect unique affected shard IDs
remove requested records from those shards
retrain only affected shards
leave all other shard models unchanged
replace affected model artifacts