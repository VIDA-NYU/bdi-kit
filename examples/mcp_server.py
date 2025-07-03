from bdikit.mcp import server
# Run the MCP server
# This will start the server and listen for requests on standard input/output. 
# So, you need a client to send requests to this server.
# The server will handle schema matching, value matching, and other tasks as defined in the tools
server.run(transport="stdio")