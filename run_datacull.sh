docker build --tag league-oracle-data-cull -f Dockerfile.cull .
docker run -d --name leagueCull --restart always --network="host" league-oracle-data-cull
