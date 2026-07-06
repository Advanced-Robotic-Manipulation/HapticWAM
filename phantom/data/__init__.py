"""phantom.data — the episode data layer (pipeline.md §1/§7).

Modules (import them directly; this package init stays dependency-light so the
recording stack works without torch):

  schema        — stream-name constants, EpisodeMeta, NormStats
  derived       — pure-numpy derived contact channels (mask/CoP/slip/events)
  episode_store — zarr-backed EpisodeWriter / EpisodeReader / list_episodes
  windows       — WindowSampler: episodes -> model window dicts (needs torch)
  synthetic     — SyntheticEpisodeGenerator (mock-scenario-scripted episodes)
  contact_play  — ContactPlayDataset for the SSL tactile pretrain (needs torch)
"""
