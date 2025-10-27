window.startSSEPost = async function (url, reqJson, dotnetRef) {
    console.log("Starting SSE POST", url);

    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: reqJson
    });

    if (!response.ok) {
        console.error("SSE request failed:", response.status);
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        let parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
            if (part.startsWith("data:")) {
                const data = part.slice(5).trim();
                try {
                    const json = JSON.parse(data);
                    console.log("SSE event:", json);
                    dotnetRef.invokeMethodAsync("ReceiveSSEMessage", json);
                } catch (e) {
                    console.error("Failed to parse SSE data", data, e);
                }
            }
        }
    }

    console.log("SSE stream closed");
};
