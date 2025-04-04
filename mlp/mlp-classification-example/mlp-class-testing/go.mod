module mlp-class-testing

go 1.24.0

require (
	my-go-packages/golang/logger v1.0.0
	my-go-packages/neural-networks/mlp v0.0.1
)

replace my-go-packages/neural-networks/mlp => /home/jeff/golang/my-go-packages/neural-networks/mlp

replace my-go-packages/golang/logger => /home/jeff/golang/my-go-packages/golang/logger
