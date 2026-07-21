integration-tests:
	cargo test

cuda-integration-tests:
	cargo test -F text-embeddings-backend-chalice/cuda -F text-embeddings-backend-chalice/flash-attn -F text-embeddings-router/chalice-cuda --profile release-debug

integration-tests-review:
	cargo insta test --review

cuda-integration-tests-review:
	cargo insta test --review --features "text-embeddings-backend-chalice/cuda text-embeddings-backend-chalice/flash-attn text-embeddings-router/chalice-cuda" --profile release-debug
