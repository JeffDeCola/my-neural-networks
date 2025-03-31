module mlp-class-testing

go 1.24.0

require (
	my-go-packages/neural-networks/mlp v0.0.1
    my-go-packages/golang/logger v0.0.2
)

replace my-go-packages/neural-networks/mlp => /home/jeff/golang/my-go-packages/neural-networks/mlp
replace my-go-packages/golang/logger => /home/jeff/golang/my-go-packages/golang/logger
