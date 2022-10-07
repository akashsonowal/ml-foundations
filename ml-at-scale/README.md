# Large Scale Machine Learning

## Data Support
1. Data Ingestion
- Streaming Ingestion: Click Stream (millions/sec)
  Clickstream is user interactions with some interface: Mobile App, Voice Assistant, Desktop
  Clickstream Tools (Pub/Sub): Kafka (Apache open source), Kinesis (AWS Managed)
  
  - install kafka
  - setup zookeeper setup (directory and port) and run the zookeeper server.
  - setup broker and run the broker server
  
  Note: while setting up you would also set topics
  
  setup a producer:
  - webapp with click stream
  - connect the falsk app with kafka broker
  - send the response from web app to kafka broker to specific topic in broker
  
  You can setup a consumer message that validates whether we are able to receive message from producer in terminal.
  
  Notes:
  I Clickstream
An ordered series of interactions that users have with some interface. In the traditional sense, this can be literal clicks of a mouse on a
desktop browser. Interactions can also come from touchscreens and conversational user interfaces.
| Change Data Capture
The process of recording changes In the data within a database system. For Instance, If a user cancels their Netflix subscription, then
the row In some table will change to indicate that they're no longer a subscriber.
The change In this row can be recorded and referenced later for analysis or audit purposes.
| Apache Kafka %
An open-source software platform which provides a way to handle real-time data streaming.
| Amazon Kinesis %
An AWS product that provides a way to handle real-time data streaming.

| Zookeeper %
Aservice designed to reliably coordinate distributed systems via naming service, configuration management, data synchronization,
leader election, message queuing, or notification systems.

| Database
Atool used to collect and organize data. Typically, database management systems allow users to Interact with the database.

I oLtp
Online transaction processing. A system that handles (near) real-time business processes. For example, a database that maintains a
table of the users subscribed to Netflix and which Is then used to enable successful log-ins would be considered OLTP. This is opposed
to OLAP.

I oLap
online analytical processing. A system that handles the analytical processes of a business, including reporting, auditing, and business
Intelligence. For example, this may be a Hadoop cluster which maintains user subscription history for Netflix. This is opposed to OLTP.

| Availability Zone (AZ)
Typically a single data center within a region which has two or more data centers. The term "multi-AZ" implies that an application or
software resource is present across more than one AZ within a reglon. This strategy allows the software resource to continue operating,
even if a data center becomes unavailable.
  
  
- Change Data Capture
- Live Video
