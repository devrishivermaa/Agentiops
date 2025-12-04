import { EventTypes } from '../types';

export const simulatePipeline = (pipelineId: string, sendEvent: (event: any) => void) => {
  let currentTime = 0;

  const schedule = (ms: number, cb: () => void) => {
    currentTime += ms;
    setTimeout(cb, currentTime);
  };

  // 1. Pipeline Start
  schedule(0, () => sendEvent({
    event_type: EventTypes.PIPELINE_STARTED,
    pipeline_id: pipelineId,
    data: { file: "annual_report_2024.pdf" }
  }));

  // 2. Master Planning
  schedule(1000, () => sendEvent({
    event_type: EventTypes.MASTER_PLAN_GENERATING,
    pipeline_id: pipelineId,
    agent_id: "master",
    agent_type: "master"
  }));

  schedule(2500, () => sendEvent({
    event_type: EventTypes.MASTER_PLAN_GENERATED,
    pipeline_id: pipelineId,
    data: { num_submasters: 2 }
  }));

  // 3. Spawn SubMasters
  const submasters = [
    { id: "sm_1", role: "Financial Analyst", pages: [1, 5] },
    { id: "sm_2", role: "Risk Assessor", pages: [6, 10] }
  ];

  submasters.forEach((sm, idx) => {
    schedule(1000 + (idx * 500), () => sendEvent({
      event_type: EventTypes.SUBMASTER_SPAWNED,
      pipeline_id: pipelineId,
      agent_id: sm.id,
      agent_type: "submaster",
      data: { role: sm.role, page_range: sm.pages }
    }));
  });

  // 4. Spawn Workers & Process
  // SubMaster 1 Workers
  const sm1Workers = [
      { id: "w_1_1", sm: "sm_1", page: 1 },
      { id: "w_1_2", sm: "sm_1", page: 2 },
      { id: "w_1_3", sm: "sm_1", page: 3 },
  ];

  sm1Workers.forEach((w, idx) => {
      // Spawn
      schedule(500 + (idx * 300), () => sendEvent({
          event_type: EventTypes.WORKER_SPAWNED,
          pipeline_id: pipelineId,
          agent_id: w.id,
          agent_type: "worker",
          data: { submaster_id: w.sm, page: w.page }
      }));

      // Process
      schedule(1000, () => sendEvent({
        event_type: EventTypes.WORKER_PROCESSING,
        pipeline_id: pipelineId,
        agent_id: w.id
      }));

      // Complete
      schedule(2000 + (Math.random() * 2000), () => {
          sendEvent({
            event_type: EventTypes.WORKER_COMPLETED,
            pipeline_id: pipelineId,
            agent_id: w.id
          });
          
          // Update SM progress roughly
          sendEvent({
             event_type: EventTypes.SUBMASTER_PROGRESS,
             pipeline_id: pipelineId,
             agent_id: w.sm,
             data: { percent: Math.min(100, (idx + 1) * 33) }
          });
      });
  });

   // SubMaster 2 Workers (delayed)
   const sm2Workers = [
    { id: "w_2_1", sm: "sm_2", page: 6 },
    { id: "w_2_2", sm: "sm_2", page: 7 },
  ];

  schedule(1000, () => {}); // Padding

  sm2Workers.forEach((w, idx) => {
    schedule(500 + (idx * 300), () => sendEvent({
        event_type: EventTypes.WORKER_SPAWNED,
        pipeline_id: pipelineId,
        agent_id: w.id,
        agent_type: "worker",
        data: { submaster_id: w.sm, page: w.page }
    }));
    
    schedule(800, () => sendEvent({
        event_type: EventTypes.WORKER_PROCESSING,
        pipeline_id: pipelineId,
        agent_id: w.id
    }));

    schedule(3000, () => {
        sendEvent({
            event_type: EventTypes.WORKER_COMPLETED,
            pipeline_id: pipelineId,
            agent_id: w.id
        });
        sendEvent({
             event_type: EventTypes.SUBMASTER_PROGRESS,
             pipeline_id: pipelineId,
             agent_id: w.sm,
             data: { percent: 100 } // Shortcuts for demo
        });
    });
  });

  // 5. Complete SubMasters
  schedule(4000, () => {
      submasters.forEach(sm => {
        sendEvent({
            event_type: EventTypes.SUBMASTER_COMPLETED,
            pipeline_id: pipelineId,
            agent_id: sm.id
        });
      });
  });

  // 6. Complete Pipeline
  schedule(2000, () => {
      sendEvent({
          event_type: EventTypes.PIPELINE_COMPLETED,
          pipeline_id: pipelineId,
          data: {}
      });
  });
};