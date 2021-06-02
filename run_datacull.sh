docker build --tag league-oracle-data-cull -f Dockerfile.cull .
docker stop leagueCull
docker rm leagueCull
docker run -d --name leagueCull --restart always --network="host" league-oracle-data-cull
