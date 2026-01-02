import asyncio
import httpx
import time

URL = "http://localhost:3000/embeddings"

async def send_request(client: httpx.AsyncClient, idx: int):
    payload = {
        "requests": [
            {
            "input": f"xin chào việt nam tôi là một thanh niên {idx}_{idx} đẹp zaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiidddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddlahkdahsjdahduecnnadagdjkadkagggeualdnnđ",
            "normalize_embeddings": True
            }
        ]
    }

    start = time.time()
    resp = await client.post(URL, json=payload)
    elapsed = time.time() - start

    data = resp.json()
    print(
        f"[REQ {idx}] status={resp.status_code} "
        f"time={elapsed:.3f}s "
        # f"emb_dim={len(data[0]['embeddings'][0])}"
        f"number of respone={len(data)}\n"
        f"data={data[0]}"
        f"\n\n==========================\n\n"

    )
    return data


async def main():
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [
            send_request(client, i)
            for i in range(20)   
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
